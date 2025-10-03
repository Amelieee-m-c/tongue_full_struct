
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

DEFAULT_LABELS = [
    "TonguePale", "TipSideRed", "Spot", "Ecchymosis",
    "Crack", "Toothmark", "FurThick", "FurYellow"
]

def multilabel_row_to_class_index(row, label_cols):
    """Pick the first label==1 as the single ground-truth class.
    Returns -1 if all are zero (row has no positive labels)."""
    for i, col in enumerate(label_cols):
        try:
            if int(row[col]) == 1:
                return i
        except Exception:
            try:
                if float(row[col]) >= 0.5:
                    return i
            except Exception:
                pass
    return -1

def load_ground_truth(gt_csv, id_col, label_cols):
    df = pd.read_csv(gt_csv)
    for c in [id_col] + label_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {gt_csv}")
    y_true = df.apply(lambda r: multilabel_row_to_class_index(r, label_cols), axis=1).values
    ids = df[id_col].values
    mask = y_true >= 0
    if (~mask).sum() > 0:
        print(f"[Info] Skipped {(~mask).sum()} samples with no positive labels.", file=sys.stderr)
    return ids[mask], y_true[mask]

def infer_predictions_from_csv(pred_csv, ids, label_cols):
    pdf = pd.read_csv(pred_csv)
    if 'id' not in pdf.columns:
        raise ValueError("Prediction CSV must contain an 'id' column.")
    pdf = pdf.set_index('id').reindex(ids)
    if pdf.isnull().any().any():
        missing = pdf.index[pdf.isnull().any(axis=1)].tolist()
        raise ValueError(f"Missing predictions for ids: {missing[:10]} (and possibly more).")

    if 'pred_label' in pdf.columns:
        name_to_idx = {name: i for i, name in enumerate(label_cols)}
        try:
            return pdf['pred_label'].map(name_to_idx).astype(int).values
        except Exception:
            raise ValueError("Could not map 'pred_label' to known label names.")

    onehot_cols = [f"pred_{c}" for c in label_cols]
    if all(c in pdf.columns for c in onehot_cols):
        arr = (pdf[onehot_cols].values > 0.5).astype(int)
        return np.array([row.argmax() if row.sum() == 0 else int(np.where(row==1)[0][0]) for row in arr], dtype=int)

    prob_cols = [f"predprob_{c}" for c in label_cols]
    if all(c in pdf.columns for c in prob_cols):
        arr = pdf[prob_cols].values.astype(float)
        return arr.argmax(axis=1).astype(int)

    raise ValueError(
        "Could not infer prediction format. "
        "Provide 'pred_label' or 'pred_<name>' or 'predprob_<name>' columns."
    )

def plot_confusion_matrix(cm, class_names, out_png, normalize=False, title=None):
    if normalize:
        with np.errstate(all='ignore'):
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Build an 8x8 confusion matrix from multi-label CSVs.")
    parser.add_argument("--gt_csv", type=str, required=True, help="Ground-truth CSV path (e.g., val_fold1.csv)")
    parser.add_argument("--pred_csv", type=str, default=None, help="Optional prediction CSV path")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "pred_labels", "pred_probs", "auto"],
                        help=("How to get predictions: 'random' for demo; "
                              "'pred_labels' expects 'pred_<name>' one-hot or 'pred_label'; "
                              "'pred_probs' expects 'predprob_<name>'; "
                              "'auto' tries to infer format."))
    parser.add_argument("--id_col", type=str, default="id", help="ID column name to join on")
    parser.add_argument("--labels", type=str, nargs="*", default=DEFAULT_LABELS, help="Label column names (8 classes)")
    parser.add_argument("--out_png", type=str, default="confusion_matrix.png", help="Output image path")
    parser.add_argument("--normalize", action="store_true", help="Output normalized percentages")
    args = parser.parse_args()

    label_cols = args.labels
    if len(label_cols) != 8:
        print(f"[Warn] You provided {len(label_cols)} labels; this script expects 8.", file=sys.stderr)

    ids, y_true = load_ground_truth(args.gt_csv, args.id_col, label_cols)

    if args.mode == "random" or (args.pred_csv is None and args.mode != "auto"):
        rng = np.random.default_rng(42)
        y_pred = rng.integers(low=0, high=len(label_cols), size=len(y_true))
        title = "Confusion Matrix (random predictions)"
    else:
        if args.pred_csv is None and args.mode == "auto":
            raise ValueError("In 'auto' mode you must provide --pred_csv.")
        if args.mode in ["pred_probs", "pred_labels", "auto"]:
            y_pred = infer_predictions_from_csv(args.pred_csv, ids, label_cols)
            title = f"Confusion Matrix ({args.mode})"
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_cols))))
    plot_confusion_matrix(cm, class_names=label_cols, out_png=args.out_png,
                          normalize=args.normalize, title=title)

if __name__ == "__main__":
    main()
