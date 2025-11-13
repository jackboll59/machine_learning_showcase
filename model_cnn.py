"""Training goal: Given an initial window of perc_change_from_entry and metadata,
predict the probability that price will rise by TARGET_GAIN and sustain it for
MIN_HOLD_TICKS after the observation window."""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc

TRACKING_FILE = "data/watch_tracking.csv"
PRICE_HISTORY_FILE = "data/price_history.csv"

INITIAL_WINDOW_TICKS = 240  # 4 minutes if 1 tick = 1 sec
MIN_HOLD_TICKS = 10  # 10 seconds of sustained gain
TARGET_GAIN = 5.0
META_FEATURES = ['liq', 'mcap', 'vol', 'm5', 'h1', 'h6', 'h24']
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 10
GAMMA = 2.0  # focal-loss gamma
LR = 0.001
DROP_RATE = 0.3
FILTERS1, FILTERS2 = 32, 64
WEIGHT_DECAY = 0.0
KERNEL_SIZE = 5       # kernel size used in conv layers & residual blocks

TRACKING_COLUMNS = {
    'perc_change': 0, 'high_perc': 1, 'low_perc': 2, 'age': 3, 'liq': 4, 'mcap': 5,
    'vol': 6, 'm5': 7, 'h1': 8, 'h6': 9, 'h24': 10, 'watch_start_time': 11,
    'watch_end_time': 12, 'watch_id': 13
}

PRICE_COLUMNS = {
    'watch_id': 0, 'timestamp': 1, 'price': 2, 'perc_change_from_entry': 3
}

def load_metadata(tracking_filename: str) -> Dict[str, List[float]]:
    metadata = {}
    with open(tracking_filename, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                watch_id = row[TRACKING_COLUMNS['watch_id']].strip()
                meta_values = [float(row[TRACKING_COLUMNS[f]]) if row[TRACKING_COLUMNS[f]] else 0.0 for f in META_FEATURES]
                metadata[watch_id] = meta_values
            except (ValueError, IndexError):
                continue
    return metadata

def compute_rsi(series, window=14):
    series = np.array(series)
    deltas = np.diff(series)
    seed = deltas[:window]
    up = seed[seed > 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down if down != 0 else 0
    rsi = 100. - 100. / (1. + rs)
    return rsi if not np.isnan(rsi) else 50.0

def collect_training_data(price_filename: str, metadata: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    watch_data = defaultdict(list)
    with open(price_filename, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                watch_id = row[PRICE_COLUMNS['watch_id']].strip()
                perc_change = float(row[PRICE_COLUMNS['perc_change_from_entry']])
                if watch_id in metadata:
                    watch_data[watch_id].append(perc_change)
            except (ValueError, IndexError):
                continue

    X, y = [], []
    for watch_id, perc_list in watch_data.items():
        if len(perc_list) <= INITIAL_WINDOW_TICKS + MIN_HOLD_TICKS:
            continue

        input_window = perc_list[:INITIAL_WINDOW_TICKS]
        obs_end = perc_list[INITIAL_WINDOW_TICKS - 1]
        future_window = perc_list[INITIAL_WINDOW_TICKS:]

        threshold = obs_end + TARGET_GAIN
        success = any(
            all(p >= threshold for p in future_window[i:i + MIN_HOLD_TICKS])
            for i in range(len(future_window) - MIN_HOLD_TICKS + 1)
        )

        target = 1 if success else 0

        mom = input_window[-1] - input_window[0]
        vol = np.std(input_window)
        rsi = compute_rsi(input_window[-15:]) if len(input_window) >= 15 else 50.0

        features = np.array(input_window).reshape(1, -1)
        meta_features = np.array(metadata[watch_id] + [mom, vol, rsi]).reshape(1, -1)

        X.append((features, meta_features))
        y.append(target)
    return X, np.array(y)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size: int = KERNEL_SIZE):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        )

    def forward(self, x):
        return torch.relu(self.block(x) + x)

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: (batch, seq, dim)
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.mean(dim=1)  # Reduce to (batch, dim)

class CNNWithAttention(nn.Module):
    def __init__(self, seq_len, meta_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, FILTERS1, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2),
            nn.ReLU(),
            ResidualBlock(FILTERS1, KERNEL_SIZE),
            nn.MaxPool1d(2),
            nn.Conv1d(FILTERS1, FILTERS2, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2),
            nn.ReLU(),
            ResidualBlock(FILTERS2, KERNEL_SIZE),
            nn.MaxPool1d(2),
        )

        self.attn = SelfAttention(FILTERS2)

        self.classifier = nn.Sequential(
            nn.Linear(FILTERS2 + meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROP_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logits output, sigmoid applied later as needed
        )

    def forward(self, x_seq, x_meta):
        x_seq = self.feature_extractor(x_seq)
        x_seq = x_seq.permute(0, 2, 1)
        x_seq = self.attn(x_seq)
        x = torch.cat([x_seq, x_meta], dim=1)
        return self.classifier(x)

def focal_loss(logits, targets, alpha=1.0, gamma=GAMMA):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits.detach())
    pt = torch.where(targets == 1, probs, 1 - probs)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()

def train_model():
    metadata = load_metadata(TRACKING_FILE)
    raw_data, y = collect_training_data(PRICE_HISTORY_FILE, metadata)

    X_seq = np.array([item[0] for item in raw_data])  # shape (n, 1, seq_len)
    X_meta = np.array([item[1] for item in raw_data])  # shape (n, meta_dim)

    valid_mask = np.isfinite(X_seq).all(axis=(1,2)) & np.isfinite(X_meta).all(axis=(1,2))
    X_seq, X_meta, y = X_seq[valid_mask], X_meta[valid_mask], y[valid_mask]

    seq_scaler = StandardScaler()
    meta_scaler = StandardScaler()
    X_seq = seq_scaler.fit_transform(X_seq.reshape(len(X_seq), -1)).reshape(len(X_seq), 1, -1)
    X_meta = meta_scaler.fit_transform(X_meta.reshape(len(X_meta), -1))
    joblib.dump((seq_scaler, meta_scaler), 'cnn_scalers.pkl')

    n = len(X_seq)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    data = torch.utils.data.TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(X_meta, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(0, train_end)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(train_end, val_end)), batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(val_end, n)), batch_size=BATCH_SIZE)

    model = CNNWithAttention(seq_len=X_seq.shape[2], meta_dim=X_meta.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (xb_seq, xb_meta, yb) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(xb_seq, xb_meta).squeeze()
            loss = focal_loss(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb_seq, xb_meta, yb in val_loader:
                logits = model(xb_seq, xb_meta).squeeze()
                val_loss += focal_loss(logits, yb).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_model_state)
    model.eval()

    y_pred, y_score, y_true = [], [], []
    with torch.no_grad():
        for xb_seq, xb_meta, yb in test_loader:
            logits = model(xb_seq, xb_meta).squeeze()
            probs = torch.sigmoid(logits)
            threshold = 0.55
            preds = (probs > threshold).float()
            y_pred.append(preds)
            y_score.append(probs)
            y_true.append(yb)

    y_pred = torch.cat(y_pred).numpy()
    y_score = torch.cat(y_score).numpy()
    y_true = torch.cat(y_true).numpy()

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Probability stats - mean: {y_score.mean():.4f}, min: {y_score.min():.4f}, max: {y_score.max():.4f}")

    ap_score = average_precision_score(y_true, y_score)
    print(f"Average Precision Score: {ap_score:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    print(f"ROC AUC Score: {auc(fpr, tpr):.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    print(f"PR AUC Score: {auc(recall, precision):.4f}")

    total = len(y_true)
    print(f"\nCorrect Predictions: {(y_pred == y_true).sum()} / {total} ({(y_pred == y_true).mean():.1%})")

    torch.save(model.state_dict(), "cnn_model.pt")
    print("Model saved to cnn_model.pt")

    results = {
        "model": "CNN",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision": float(ap_score),
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
        "prob_mean": float(y_score.mean()),
        "prob_min": float(y_score.min()),
        "prob_max": float(y_score.max()),
        "correct": int((y_pred == y_true).sum()),
        "total": int(total),
    }
    return results

def main():
    train_model()

if __name__ == "__main__":
    main()
