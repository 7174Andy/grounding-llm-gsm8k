import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        return pd.read_json(file_path, lines=True)
    except ValueError:
        return pd.read_json(file_path, lines=False)


def train_one_layer(X, y, C=1.0, max_iter=1000):
    clf = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs",
        max_iter=max_iter, class_weight="balanced"
    )
    clf.fit(X, y)
    return clf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='ID of the model to train')
    args = parser.parse_args()

    # Extract labels
    model_name_for_dir = args.model_id.split("/")[-1]
    data_path = Path(f"{model_name_for_dir}_dataset")
    predictions = load_data(data_path / "baseline_predictions.json")
    y = predictions["label"].astype(int).to_numpy()

    # Extract layer files
    layer_files = sorted(data_path.glob("layer_*.npy"))
    if not layer_files:
        raise ValueError("No layer files found")
    N = np.load(layer_files[0]).shape[0]
    if len(y) != N:
        raise ValueError(f"Number of labels {len(y)} does not match number of samples {N}")

     # Fixed split
    idx = np.arange(len(y))
    tr_idx, va_idx, y_tr, y_va = train_test_split(
        idx, y, test_size=args.val_size, random_state=42, stratify=y
    )

    cav_dir = data_path / "CAV"
    cav_dir.mkdir(exist_ok=True)

    results = []
    for layer in layer_files:
        X = np.load(layer)
        clf = train_one_layer(X[tr_idx], y_tr)
        X_va = X[va_idx]
        proba = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba)
        acc = accuracy_score(y_va, proba > 0.5)

        w = clf.coef_.ravel()
        v = w / (np.linalg.norm(w) + 1e-12)
        # flip so positives score higher
        if (X[va_idx][y_va==1] @ v).mean() < (X[va_idx][y_va==0] @ v).mean():
            v = -v

        np.save(cav_dir / layer.name.replace("layer_", "cav_layer_"), v.astype(np.float32))
        results.append({"layer": layer.name, "auc": float(auc), "acc": float(acc)})

    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(cav_dir / "results.csv")
    print("CAV training complete. Results saved to:", cav_dir / "results.csv")

if __name__ == "__main__":
    main()