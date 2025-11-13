"""Training goal: Using last LOOKBACK_TICKS of perc_change_from_entry plus momentum
and metadata, classify whether perc_change_from_entry will increase over the next
FORECAST_HORIZON ticks (directional up move)."""

import csv
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from typing import Dict, List, Tuple, Generator
import gc

TRACKING_FILE = "data/watch_tracking.csv"
PRICE_HISTORY_FILE = "data/price_history.csv"

LOOKBACK_TICKS = 30
FORECAST_HORIZON = 10
MIN_SESSION_LENGTH = LOOKBACK_TICKS + FORECAST_HORIZON + 5

TRACKING_COLUMNS = {
    'perc_change': 0,
    'high_perc': 1, 
    'low_perc': 2,
    'age': 3,
    'liq': 4,
    'mcap': 5,
    'vol': 6,
    'm5': 7,
    'h1': 8,
    'h6': 9,
    'h24': 10,
    'watch_start_time': 11,
    'watch_end_time': 12,
    'watch_id': 13
}

PRICE_COLUMNS = {
    'watch_id': 0,
    'timestamp': 1,
    'price': 2,
    'perc_change_from_entry': 3
}

META_FEATURES = ['liq', 'mcap', 'vol', 'm5', 'h1', 'h6', 'h24']

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

                    ts_raw = row[TRACKING_COLUMNS['watch_start_time']].strip()
                    try:
                        ts_int = int(float(ts_raw))
                        dt_obj = datetime.utcfromtimestamp(ts_int)
                    except (ValueError, OSError):
                        try:
                            dt_obj = datetime.fromisoformat(ts_raw)
                        except ValueError:
                            dt_obj = None

                    if dt_obj is not None:
                        dow = float(dt_obj.weekday())  # 0=Mon
                        hour = float(dt_obj.hour)
                    else:
                        dow = 0.0
                        hour = 0.0

                    meta_values.extend([dow, hour])

                    metadata[watch_id] = meta_values
                except (ValueError, IndexError):
                    continue
    return metadata

def process_price_history_chunk(price_filename: str, metadata: Dict[str, List[float]], chunk_size: int = 50000) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Stream raw tick records and build training samples without loading all data into memory.
    - Accumulates perc_change_by_watch in a dict
    - Periodically yields feature/target samples via _generate_samples_from_watch_data
    - Uses sliding windows of LOOKBACK_TICKS and FORECAST_HORIZON
    This reduces peak memory and keeps processing scalable for large CSVs.
    """
    watch_data = defaultdict(list)
    processed_records = 0

    with open(price_filename, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            if row and len(row) > 3:
                try:
                    watch_id = row[PRICE_COLUMNS['watch_id']].strip()
                    perc_change = float(row[PRICE_COLUMNS['perc_change_from_entry']])
                    if watch_id in metadata:
                        watch_data[watch_id].append(perc_change)
                        processed_records += 1
                        if processed_records % chunk_size == 0:
                            for sample in _generate_samples_from_watch_data(watch_data, metadata):
                                yield sample
                            watch_data = defaultdict(list)
                            gc.collect()
                except (ValueError, IndexError):
                    continue
        for sample in _generate_samples_from_watch_data(watch_data, metadata):
            yield sample

def _generate_samples_from_watch_data(watch_data: Dict[str, List[float]], metadata: Dict[str, List[float]]) -> Generator[Tuple[np.ndarray, int], None, None]:
    for watch_id, price_changes in watch_data.items():
        if len(price_changes) < MIN_SESSION_LENGTH:
            continue
        meta_features = metadata[watch_id]
        for i in range(LOOKBACK_TICKS, len(price_changes) - FORECAST_HORIZON):
            past_ticks = price_changes[i - LOOKBACK_TICKS:i]

            momentum = past_ticks[-1] - past_ticks[0]

            current_value = price_changes[i]
            future_value = price_changes[i + FORECAST_HORIZON]

            target = 1 if future_value > current_value else 0
            features = np.concatenate([past_ticks, [momentum], meta_features])
            yield (features, target)

def collect_training_data(price_filename: str, metadata: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for features, target in process_price_history_chunk(price_filename, metadata):
        X_list.append(features)
        y_list.append(target)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

def train_model():
    tracking_file = TRACKING_FILE
    price_history_file = PRICE_HISTORY_FILE

    print("Random Forest Classification Model")
    print(f"Lookback ticks: {LOOKBACK_TICKS}")
    print(f"Forecast horizon: {FORECAST_HORIZON}")
    print(f"Meta features: {META_FEATURES}")
    
    metadata = load_metadata(tracking_file)
    if not metadata:
        print("No metadata loaded. Check tracking file.")
        return

    X, y = collect_training_data(price_history_file, metadata)
    if len(X) == 0:
        print("No training samples generated. Check data files.")
        return

    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print()
    
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Model performance")
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    y_score = model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
    ap_score = average_precision_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    print(f"Probability stats - mean: {y_score.mean():.4f}, min: {y_score.min():.4f}, max: {y_score.max():.4f}")
    print(f"Average Precision Score: {ap_score:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"PR AUC Score: {pr_auc:.4f}")

    feature_names = [f'tick_{i+1}' for i in range(LOOKBACK_TICKS)] + ['momentum'] + META_FEATURES + ['day_of_week', 'hour_of_day']
    importances = model.feature_importances_
    print("Feature importance")
    for name, importance in zip(feature_names, importances):
        print(f"{name:15s}: {importance:.4f}")

    correct = int((y_pred == y_test).sum())
    total = int(len(y_test))
    if y_score is not None:
        results = {
            "model": "RandomForest",
            "accuracy": float(acc),
            "average_precision": float(ap_score),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "prob_mean": float(y_score.mean()),
            "prob_min": float(y_score.min()),
            "prob_max": float(y_score.max()),
            "correct": correct,
            "total": total,
        }
    else:
        results = {
            "model": "RandomForest",
            "accuracy": float(acc),
            "correct": correct,
            "total": total,
        }
    return results

def main():
    try:
        train_model()
        print("Model training completed successfully.")
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
