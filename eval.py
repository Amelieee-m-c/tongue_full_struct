
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DEFAULT_LABELS = [
    "TonguePale", "TipSideRed", "Spot", "Ecchymosis",
    "Crack", "Toothmark", "FurThick", "FurYellow"
]

def multilabel_to_class_index(row: pd.Series, label_cols: List[str]) -> int:
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

class TongueDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, label_cols: List[str], id_col: str="id", image_col: str="image_path", img_size: int=224):
        df = pd.read_csv(csv_path)
        needed = [id_col, image_col] + label_cols
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        ids, gts_mc, gts_ml, paths = [], [], [], []
        for _, r in df.iterrows():
            cls = multilabel_to_class_index(r, label_cols)
            if cls < 0:
                continue
            ids.append(int(r[id_col]))
            gts_mc.append(cls)
            gts_ml.append([int(r[c]) for c in label_cols])
            paths.append(str(r[image_col]))

        self.ids = np.array(ids, dtype=int)
        self.y_true_mc = np.array(gts_mc, dtype=int)
        self.y_true_ml = np.array(gts_ml, dtype=int)
        self.paths = np.array(paths, dtype=str)
        self.img_root = img_root
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        p = os.path.join(self.img_root, self.paths[idx])
        with Image.open(p) as im:
            im = im.convert("RGB")
        x = self.tf(im)
        return x, self.y_true_mc[idx], self.y_true_ml[idx], self.ids[idx]

def build_resnet18(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_model(weights: str, arch: str, num_classes: int, custom_py: Optional[str]=None, custom_build_fn: Optional[str]=None, device: str="cpu"):
    arch = arch.lower()
    if arch == "resnet18":
        model = build_resnet18(num_classes)
        ckpt = torch.load(weights, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError("Unsupported checkpoint structure for resnet18 state_dict.")
        model.to(device).eval()
        return model, False
    elif arch == "torchscript":
        model = torch.jit.load(weights, map_location=device)
        model.eval()
        return model, True
    elif arch == "custom":
        if not custom_py or not custom_build_fn:
            raise ValueError("For arch=custom, please provide --custom_py and --custom_build_fn")
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_model_module", custom_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, custom_build_fn):
            raise ValueError(f"{custom_py} does not define function '{custom_build_fn}'")
        build_fn = getattr(mod, custom_build_fn)
        model = build_fn(num_classes=num_classes)
        ckpt = torch.load(weights, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError("Unsupported checkpoint structure for custom model.")
        model.to(device).eval()
        return model, False
    else:
        raise ValueError(f"Unknown arch: {arch}")

@torch.no_grad()
def run_inference(model, loader: DataLoader, scripted: bool, device: str):
    all_ids, y_true_mc, y_pred_mc, y_true_ml, y_prob = [], [], [], [], []
    for x, t_mc, t_ml, _ids in loader:
        x = x.to(device)
        logits = model(x) if scripted else model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        y_prob.append(probs)
        y_pred_mc.append(pred)
        y_true_mc.append(t_mc.numpy())
        y_true_ml.append(t_ml.numpy())
        all_ids.append(_ids.numpy())
    return (np.concatenate(all_ids),
            np.concatenate(y_true_mc),
            np.concatenate(y_pred_mc),
            np.concatenate(y_true_ml),
            np.concatenate(y_prob))

def plot_cm(cm: np.ndarray, class_names: List[str], out_png: str, normalize: bool = False, title: str = None):
    if normalize:
        with np.errstate(all="ignore"):
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, aspect="auto")
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
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def per_label_confusion_matrices(y_true_ml: np.ndarray, y_prob: np.ndarray, out_dir: str, class_names: List[str], threshold: float=0.5):
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(class_names):
        y_true = y_true_ml[:, i].astype(int)
        y_pred = (y_prob[:, i] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        out_png = str(Path(out_dir) / f"cm_binary_{i:02d}_{name}.png")
        plot_cm(cm, class_names=["0", "1"], out_png=out_png, normalize=False, title=f"{name} (binary CM)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv",   type=str, required=True)
    ap.add_argument("--img_root",  type=str, required=True)
    ap.add_argument("--weights",   type=str, required=True)
    ap.add_argument("--arch",      type=str, default="resnet18", choices=["resnet18","torchscript","custom"])
    ap.add_argument("--custom_py", type=str, default=None)
    ap.add_argument("--custom_build_fn", type=str, default="build_model")
    ap.add_argument("--labels",    type=str, nargs="*", default=DEFAULT_LABELS)
    ap.add_argument("--out_dir",   type=str, default="./runs/eval")
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--img_size",  type=int, default=224)
    ap.add_argument("--normalize_cm", action="store_true")
    ap.add_argument("--save_binary_cm", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ds = TongueDataset(args.val_csv, args.img_root, args.labels, img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model, scripted = load_model(args.weights, args.arch, num_classes=len(args.labels),
                                 custom_py=args.custom_py, custom_build_fn=args.custom_build_fn, device=device)

    ids, y_true_mc, y_pred_mc, y_true_ml, y_prob = run_inference(model, loader, scripted, device)

    pred_names = [args.labels[i] for i in y_pred_mc]
    prob_cols = {f"predprob_{n}": y_prob[:, i] for i, n in enumerate(args.labels)}
    df = pd.DataFrame({"id": ids, "pred_index": y_pred_mc, "pred_label": pred_names})
    for k, v in prob_cols.items():
        df[k] = v
    df.to_csv(str(Path(args.out_dir) / "val_predictions.csv"), index=False)

    cm = confusion_matrix(y_true_mc, y_pred_mc, labels=list(range(len(args.labels))))
    plot_cm(cm, class_names=args.labels, out_png=str(Path(args.out_dir) / "val_confusion_matrix.png"),
            normalize=args.normalize_cm, title="Validation Confusion Matrix")

    if args.save_binary_cm:
        per_label_confusion_matrices(y_true_ml, y_prob, out_dir=str(Path(args.out_dir) / "binary_cms"),
                                     class_names=args.labels, threshold=args.threshold)

    print("[Done] Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
