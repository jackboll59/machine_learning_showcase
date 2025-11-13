"""Training goal: Given an initial window of perc_change_from_entry and metadata,
predict the probability that price will rise by TARGET_GAIN and sustain it for
MIN_HOLD_TICKS after the observation window (aligned with cnn.py)."""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
from collections import defaultdict
from typing import Dict, List, Tuple

TRACKING_FILE = "data/watch_tracking.csv"
PRICE_HISTORY_FILE = "data/price_history.csv"

INITIAL_WINDOW_TICKS = 240
MIN_HOLD_TICKS = 10
TARGET_GAIN = 5.0
META_FEATURES = ['liq', 'mcap', 'vol', 'm5', 'h1', 'h6', 'h24']
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 10

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
        header = next(reader)
        for row in reader:
            if row and len(row) > 13:
                try:
                    watch_id = row[TRACKING_COLUMNS['watch_id']].strip()
                    meta_values = [float(row[TRACKING_COLUMNS[f]]) if row[TRACKING_COLUMNS[f]] else 0.0 for f in META_FEATURES]
                    metadata[watch_id] = meta_values
                except (ValueError, IndexError):
                    continue
    return metadata

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

        mom = input_window[-1] - input_window[0]
        vol = np.std(input_window)
        rsi = _compute_rsi(input_window[-15:]) if len(input_window) >= 15 else 50.0

        flat_features = np.array(list(input_window) + metadata[watch_id] + [mom, vol, rsi], dtype=float)
        X.append(flat_features)
        y.append(1 if success else 0)

    return np.array(X), np.array(y)

def _compute_rsi(series, window=14):
    series = np.array(series)
    deltas = np.diff(series)
    seed = deltas[:window]
    up = seed[seed > 0].sum() / window if window <= len(seed) else 0.0
    down = -seed[seed < 0].sum() / window if window <= len(seed) else 0.0
    rs = up / down if down != 0 else 0.0
    rsi = 100. - 100. / (1. + rs) if rs != 0 else 50.0
    return rsi if not np.isnan(rsi) else 50.0

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_model():
    metadata = load_metadata(TRACKING_FILE)
    X, y = collect_training_data(PRICE_HISTORY_FILE, metadata)
    valid_mask = np.isfinite(X).all(axis=1)
    X, y = X[valid_mask], y[valid_mask]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, 'mlp_scaler.pkl')

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_tensor = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_tensor = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_tensor = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=BATCH_SIZE)

    class_weights = torch.tensor([1.0, (y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
    criterion = nn.BCELoss(reduction='none')

    model = MLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            weights = torch.where(yb == 1, class_weights[1], class_weights[0])
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                weights = torch.where(yb == 1, class_weights[1], class_weights[0])
                val_loss += (loss * weights).mean().item()

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

    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = model(xb).squeeze()
            preds = (probs > 0.5).float()
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_score = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Probability stats - mean: {y_score.mean():.4f}, min: {y_score.min():.4f}, max: {y_score.max():.4f}")

    ap_score = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    print(f"Average Precision Score: {ap_score:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"PR AUC Score: {pr_auc:.4f}")

    total = len(y_true)
    print(f"\nCorrect Predictions: {(y_pred == y_true).sum()} / {total} ({(y_pred == y_true).mean():.1%})")

    torch.save(model.state_dict(), "mlp_model.pt")
    print("Model saved to mlp_model.pt")

    results = {
        "model": "MLP",
        "accuracy": float(acc),
        "average_precision": float(ap_score),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
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
