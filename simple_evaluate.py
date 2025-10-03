import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix
)

# ------------------------
# 1. Dataset（和訓練一致）
# ------------------------
class TongueDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_cols, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def crop_parts(self, img):
        w, h = img.size
        body = img.crop((0, h // 4, w, 3 * h // 4))   # 中間部分
        edge = img.crop((w // 4, 0, 3 * w // 4, h))   # 左右裁掉
        whole = img                                    # 原圖
        return body, edge, whole

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_path"])
        img = Image.open(img_path).convert("RGB")

        body, edge, whole = self.crop_parts(img)

        if self.transform:
            body = self.transform(body)
            edge = self.transform(edge)
            whole = self.transform(whole)

        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return (body, edge, whole), labels, row["image_path"]

# ------------------------
# 2. Multi-Branch 模型（對齊訓練用 ResNet50）
# ------------------------
class MultiBranchTongueNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet50(pretrained=True)

        def branch():
            m = models.resnet50(pretrained=True)
            m.fc = nn.Identity()
            return m

        self.body_net = branch()
        self.edge_net = branch()
        self.whole_net = branch()

        feat_dim = base.fc.in_features * 3
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, inputs):
        body, edge, whole = inputs
        fb = self.body_net(body)
        fe = self.edge_net(edge)
        fw = self.whole_net(whole)

        feats = torch.cat([fb, fe, fw], dim=1).unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attn(feats, feats, feats)
        out = self.fc(attn_out.squeeze(1))
        return out

# ------------------------
# 3. 視覺化：8×8 多標籤「共現」矩陣
#    （True=1 與 Pred=1 的交叉計數）
# ------------------------
def multilabel_confusion_8x8(all_labels_bin, all_preds_bin, label_cols, save_path="confusion_matrix_8x8_multilabel.png"):
    n_labels = len(label_cols)
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for i in range(all_labels_bin.shape[0]):
        for t in range(n_labels):   # 真實標籤為 1 的類別
            if all_labels_bin[i, t] == 1:
                for p in range(n_labels):  # 預測為 1 的類別
                    if all_preds_bin[i, p] == 1:
                        cm[t, p] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_cols, yticklabels=label_cols)
    plt.title("8x8 Co-occurrence Matrix (True=1 vs Pred=1)")
    plt.xlabel("Predicted label = 1")
    plt.ylabel("True label = 1")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 已輸出 8x8 多標籤共現矩陣 -> {save_path}")

# ------------------------
# 4. 主驗證流程
# ------------------------
def validate_model(
    val_csv,
    img_dir,
    label_cols,
    model_path,
    batch_size=8,
    num_workers=4,
    base_threshold=0.5,
    out_dir="val_results",
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_ds = TongueDataset(val_csv, img_dir, label_cols, transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # 建模與載權重（對齊訓練）
    model = MultiBranchTongueNet(num_classes=len(label_cols)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_probs = []
    all_labels = []
    all_img_paths = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", unit="batch")
        for (imgs, labels, img_paths) in pbar:
            imgs = [x.to(device) for x in imgs]
            outputs = model(imgs)                        # logits
            probs = torch.sigmoid(outputs).cpu().numpy() # 轉機率

            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_img_paths.extend(list(img_paths))

    all_probs = np.concatenate(all_probs, axis=0)   # [N, C]
    all_labels = np.concatenate(all_labels, axis=0) # [N, C]

    # ------------------------
    # 4.1 計算逐類 AUROC / AP
    # ------------------------
    aurocs, aps = [], []
    for c in range(len(label_cols)):
        y_true = all_labels[:, c]
        y_score = all_probs[:, c]
        # AUROC/AP 在單一類別全 0 或全 1 時會報錯，做個保護
        if np.unique(y_true).size == 1:
            aurocs.append(np.nan)
            aps.append(np.nan)
        else:
            aurocs.append(roc_auc_score(y_true, y_score))
            aps.append(average_precision_score(y_true, y_score))

    mAP = np.nanmean(aps)
    macro_auc = np.nanmean(aurocs)

    # ------------------------
    # 4.2 逐類最佳閾值（以 F1 最大化為目標）
    # ------------------------
    best_ths = []
    for c in range(len(label_cols)):
        y_true = all_labels[:, c].astype(int)
        y_score = all_probs[:, c]
        if np.unique(y_true).size == 1:
            best_ths.append(base_threshold)  # 單類無法優化，回退到 base
            continue
        # 掃描 0.05~0.95
        ths = np.linspace(0.05, 0.95, 19)
        f1s = []
        for th in ths:
            y_pred = (y_score >= th).astype(int)
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        best_ths.append(float(ths[int(np.argmax(f1s))]))

    # 使用最佳閾值產生二值預測
    all_preds_bin = np.zeros_like(all_probs, dtype=int)
    for c, th in enumerate(best_ths):
        all_preds_bin[:, c] = (all_probs[:, c] >= th).astype(int)

    # ------------------------
    # 4.3 整體指標（micro / macro）
    # ------------------------
    prf_macro = precision_recall_fscore_support(
        all_labels.ravel(),
        all_preds_bin.ravel(),
        average="macro",
        zero_division=0
    )
    prf_micro = precision_recall_fscore_support(
        all_labels.ravel(),
        all_preds_bin.ravel(),
        average="micro",
        zero_division=0
    )

    print("\n===== Validation Metrics =====")
    print(f"mAP: {mAP:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"Macro P/R/F1: P={prf_macro[0]:.4f} R={prf_macro[1]:.4f} F1={prf_macro[2]:.4f}")
    print(f"Micro P/R/F1: P={prf_micro[0]:.4f} R={prf_micro[1]:.4f} F1={prf_micro[2]:.4f}")

    print("\nPer-label:")
    for i, name in enumerate(label_cols):
        y_true = all_labels[:, i]
        y_pred = all_preds_bin[:, i]
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        auc_i = aurocs[i]; ap_i = aps[i]
        print(f"- {name:>12s}: AUC={auc_i:.4f} AP={ap_i:.4f}  P={p:.4f} R={r:.4f} F1={f1:.4f}  thr={best_ths[i]:.2f}")

    # ------------------------
    # 4.4 輸出 256×256 組合混淆矩陣（注意：可能非常稀疏）
    # ------------------------
    def bin2int(arr_bin):
        return [int("".join(map(str, row.astype(int))), 2) for row in arr_bin]

    y_true_combo = bin2int(all_labels)
    y_pred_combo = bin2int(all_preds_bin)

    cm_combo = confusion_matrix(y_true_combo, y_pred_combo)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_combo, cmap="Blues", cbar=True)
    plt.title("256x256 Confusion Matrix (Multi-label combinations)")
    plt.xlabel("Predicted combo")
    plt.ylabel("True combo")
    plt.tight_layout()
    combo_path = os.path.join(out_dir, "confusion_matrix_256x256.png")
    plt.savefig(combo_path)
    print(f"✅ 已輸出 256x256 混淆矩陣 -> {combo_path}")

    # ------------------------
    # 4.5 8×8 多標籤共現矩陣（True=1 × Pred=1）
    # ------------------------
    co_path = os.path.join(out_dir, "confusion_matrix_8x8_multilabel.png")
    multilabel_confusion_8x8(all_labels, all_preds_bin, label_cols, save_path=co_path)

    # ------------------------
    # 4.6 逐類 2×2 混淆矩陣（可選）
    # ------------------------
    per_label_cm = {}
    for i, name in enumerate(label_cols):
        per_label_cm[name] = confusion_matrix(all_labels[:, i], all_preds_bin[:, i], labels=[0,1])

    # 簡單存成 CSV
    cm_rows = []
    for name, mat in per_label_cm.items():
        tn, fp, fn, tp = mat.ravel()
        cm_rows.append({"label": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    pd.DataFrame(cm_rows).to_csv(os.path.join(out_dir, "per_label_confusion_2x2.csv"), index=False)
    print(f"✅ 已輸出逐類 2x2 混淆矩陣 -> {os.path.join(out_dir, 'per_label_confusion_2x2.csv')}")

    # ------------------------
    # 4.7 匯出每張圖的預測（機率+二值）與最佳閾值
    # ------------------------
    prob_df = pd.DataFrame(all_probs, columns=[f"{c}_prob" for c in label_cols])
    pred_df = pd.DataFrame(all_preds_bin, columns=[f"{c}_pred" for c in label_cols])
    true_df = pd.DataFrame(all_labels, columns=[f"{c}_true" for c in label_cols])
    img_df  = pd.DataFrame({"image_path": all_img_paths})
    out_df = pd.concat([img_df, true_df, prob_df, pred_df], axis=1)
    out_csv = os.path.join(out_dir, "val_predictions.csv")
    out_df.to_csv(out_csv, index=False)

    th_df = pd.DataFrame({"label": label_cols, "best_threshold": best_ths, "AUC": aurocs, "AP": aps})
    th_df.to_csv(os.path.join(out_dir, "per_label_thresholds_auc_ap.csv"), index=False)

    print(f"✅ 已輸出驗證預測 CSV -> {out_csv}")
    print(f"✅ 已輸出逐類最佳閾值/AUC/AP -> {os.path.join(out_dir, 'per_label_thresholds_auc_ap.csv')}")

    return {
        "mAP": mAP,
        "macro_auc": macro_auc,
        "macro_PRF": prf_macro,
        "micro_PRF": prf_micro,
        "per_label_auc": aurocs,
        "per_label_ap": aps,
        "best_thresholds": best_ths,
    }

# ------------------------
# 5. 入口
# ------------------------
if __name__ == "__main__":
    LABEL_COLS = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis",
                  "Crack", "Toothmark", "FurThick", "FurYellow"]
    img_dir = "C:/Users/msp/Tongue-AI-V2/images"
    fold_id = 1  # 想驗證第幾折
    val_csv = f"val_fold{fold_id}.csv"
    model_path = f"multibranch_tongue_fold{fold_id}.pth"

    # 產物會寫到 val_results/ 底下
    validate_model(
        val_csv=val_csv,
        img_dir=img_dir,
        label_cols=LABEL_COLS,
        model_path=model_path,
        batch_size=8,
        num_workers=4,
        base_threshold=0.5,
        out_dir=f"val_results/fold{fold_id}"
    )
