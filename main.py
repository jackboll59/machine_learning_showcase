import io
import os
import sys
from datetime import datetime

import model_cnn
import model_mlp
import model_random_forest
import model_rnn_simple


def get_goal(module) -> str:
    doc = (module.__doc__ or "").strip()
    return doc.splitlines()[0] if doc else ""


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"# Running models ({timestamp})")

    summaries = []

    print("\n## MLP (mlp.py)")
    print(f"Goal: {get_goal(model_mlp)}")
    mlp_res = model_mlp.train_model()
    summaries.append(mlp_res or {"model": "MLP"})

    print("\n## Random Forest (random_forest.py)")
    print(f"Goal: {get_goal(model_random_forest)}")
    rf_res = model_random_forest.train_model()
    summaries.append(rf_res or {"model": "RandomForest"})

    print("\n## CNN (cnn.py)")
    print(f"Goal: {get_goal(model_cnn)}")
    cnn_res = model_cnn.train_model()
    summaries.append(cnn_res or {"model": "CNN"})

    print("\n## RNN (rnn_simple.py)")
    print(f"Goal: {get_goal(model_rnn_simple)}")
    rnn_res = model_rnn_simple.train_model()
    summaries.append(rnn_res or {"model": "RNN"})

    print("\n==== Final Summary ====")
    for res in summaries:
        name = res.get("model", "Model")
        parts = [name]
        if "accuracy" in res:
            parts.append(f"Acc {res['accuracy']:.4f}" if isinstance(res["accuracy"], float) else f"Acc {res['accuracy']}")
        if "average_precision" in res:
            parts.append(f"AP {res['average_precision']:.4f}")
        if "roc_auc" in res:
            parts.append(f"ROC AUC {res['roc_auc']:.4f}")
        if "pr_auc" in res:
            parts.append(f"PR AUC {res['pr_auc']:.4f}")
        if "correct" in res and "total" in res and res["total"]:
            pct = 100.0 * res["correct"] / res["total"]
            parts.append(f"Correct {res['correct']}/{res['total']} ({pct:.1f}%)")
        if "prob_mean" in res and "prob_min" in res and "prob_max" in res:
            parts.append(f"P(mean/min/max) {res['prob_mean']:.4f}/{res['prob_min']:.4f}/{res['prob_max']:.4f}")
        print(" - " + " | ".join(parts))


if __name__ == "__main__":
    main()

