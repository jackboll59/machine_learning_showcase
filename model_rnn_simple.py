"""Training goal: Sequence model that predicts, for each tick in a session,
the probability that price exceeds entry by >2% at that tick."""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

TRACKING_FILE = "data/watch_tracking.csv"
PRICE_HISTORY_FILE = "data/price_history.csv"
EPOCHS = 15
BATCH_SIZE = 1


def preprocess_sessions():
    price_df = pd.read_csv(PRICE_HISTORY_FILE)
    meta_df = pd.read_csv(TRACKING_FILE)

    price_df = price_df.merge(meta_df.drop(columns=["watch_start_time", "watch_end_time"]), on="watch_id", how="left")
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
    price_df = price_df.sort_values(["watch_id", "timestamp"])

    session_data = {}
    for watch_id, group in price_df.groupby("watch_id"):
        group = group.reset_index(drop=True)
        if len(group) < 10:
            continue

        price = group["price"]

        group["rsi"] = RSIIndicator(close=price, window=14).rsi().fillna(0)
        group["sma_10"] = SMAIndicator(close=price, window=10).sma_indicator().bfill()
        group["sma_50"] = SMAIndicator(close=price, window=50).sma_indicator().bfill()
        group["atr"] = AverageTrueRange(high=price, low=price, close=price, window=14).average_true_range().fillna(0)
        group["momentum"] = price.diff().fillna(0)

        features = group[[
            "price", "perc_change_from_entry", "age", "liq", "mcap", "vol", "m5", "h1", "h6", "h24",
            "rsi", "sma_10", "sma_50", "atr", "momentum"
        ]].astype(np.float32).values

        entry_price = group.iloc[0]["price"]
        label = ((group["price"] - entry_price) / entry_price > 0.02).astype(np.float32).values

        session_data[watch_id] = {"X": features, "y": label}

    np.savez_compressed("precomputed_sessions.npz", **session_data)
    print(f"Saved {len(session_data)} sessions.")


def train_model():

    preprocess_sessions()

    data = np.load("precomputed_sessions.npz", allow_pickle=True)

    sequence_data = []
    for watch_id in data.files:
        session = data[watch_id].item()
        X = session["X"]
        y = session["y"]
        sequence_data.append((X, y))

    all_labels = np.concatenate([y for _, y in sequence_data])
    unique, counts = np.unique(all_labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    print("Label distribution:", label_dist)

    pos_ratio = counts[1] / (counts[0] + counts[1]) if 1.0 in label_dist else 0.01
    neg_ratio = 1.0 - pos_ratio
    pos_weight = torch.tensor([neg_ratio / pos_ratio if pos_ratio > 0 else 1.0])
    print("Pos weight:", pos_weight.item())

    train_data, test_data = train_test_split(sequence_data, test_size=0.2, random_state=42)

    class WatchSessionDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            X, y = self.data[idx]
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            X = np.clip(X, -1e6, 1e6).astype(np.float32)
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            y = np.clip(y, 0, 1).astype(np.float32)
            return torch.from_numpy(X), torch.from_numpy(y)

    train_loader = DataLoader(WatchSessionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(WatchSessionDataset(test_data), batch_size=BATCH_SIZE)

    class TradingRNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            output, _ = self.rnn(x)
            return self.fc(output).squeeze(-1)

    input_dim = train_data[0][0].shape[1]
    model = TradingRNN(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    MAX_GRAD_NORM = 1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        train_correct, train_total = 0, 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            logits = model(X)

            if torch.isnan(logits).any() or torch.isnan(y).any():
                print("NaNs in logits or targets")
                continue

            loss = criterion(logits, y)
            if torch.isnan(loss):
                print("NaN in loss")
                print("Logits:", logits)
                print("Targets:", y)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (preds == y).sum().item()
                train_total += y.numel()

        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.2%}")

    model.eval()
    correct, total = 0, 0
    all_probs = []

    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            all_probs.append(probs.mean().item())

    print(f"Test Accuracy: {correct / total:.2%}")
    print(f"Average predicted confidence (sigmoid output): {np.mean(all_probs):.4f}")

    results = {
        "model": "RNN",
        "accuracy": float(correct / total),
        "prob_mean": float(np.mean(all_probs)),
        "correct": int(correct),
        "total": int(total),
    }
    return results




def main():
    train_model()


if __name__ == "__main__":
    main()
