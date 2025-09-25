import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------------
# 1. Dataset
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
        body = img.crop((0, h // 4, w, 3 * h // 4))
        edge = img.crop((w // 4, 0, 3 * w // 4, h))
        whole = img
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
        return (body, edge, whole), labels

# ------------------------
# 2. Multi-Branch 模型
# ------------------------
class MultiBranchTongueNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(pretrained=True)

        def branch():
            m = models.resnet18(pretrained=True)
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

        feats = torch.cat([fb, fe, fw], dim=1).unsqueeze(1)
        attn_out, _ = self.attn(feats, feats, feats)
        out = self.fc(attn_out.squeeze(1))
        return out

# ------------------------
# 3. 生成 8x8 多標籤混淆矩陣
# ------------------------
def multilabel_confusion_8x8(all_labels, all_preds, label_cols):
    n_labels = len(label_cols)
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for i in range(all_labels.shape[0]):
        for t in range(n_labels):   # 真實標籤
            if all_labels[i, t] == 1:
                for p in range(n_labels):  # 預測標籤
                    if all_preds[i, p] == 1:
                        cm[t, p] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_cols, yticklabels=label_cols)
    plt.title("8x8 Confusion Matrix (multi-label counting)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_8x8_multilabel.png")
    print("✅ 已輸出 8x8 多標籤混淆矩陣 -> confusion_matrix_8x8_multilabel.png")

# ------------------------
# 4. 驗證函數
# ------------------------
def validate_model(val_csv, img_dir, label_cols, model_path, batch_size=8, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_ds = TongueDataset(val_csv, img_dir, label_cols, transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiBranchTongueNet(num_classes=len(label_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = [x.to(device) for x in imgs]
            labels_np = labels.cpu().numpy()
            outputs = model(imgs)
            preds_np = (torch.sigmoid(outputs).cpu().numpy() > threshold).astype(int)

            all_labels.extend(labels_np)
            all_preds.extend(preds_np)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # ------------------------
    # 256x256 混淆矩陣
    # ------------------------
    def bin2int(arr):
        return [int("".join(map(str, row.astype(int))), 2) for row in arr]

    y_true_combo = bin2int(all_labels)
    y_pred_combo = bin2int(all_preds)

    cm_combo = confusion_matrix(y_true_combo, y_pred_combo)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_combo, cmap="Blues", cbar=True)
    plt.title("256x256 Confusion Matrix (Multi-label combination)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_256x256.png")
    print("✅ 已輸出 256x256 混淆矩陣 -> confusion_matrix_256x256.png")

    # ------------------------
    # 8x8 多標籤混淆矩陣
    # ------------------------
    multilabel_confusion_8x8(all_labels, all_preds, label_cols)

# ------------------------
# 5. 主程式
# ------------------------
if __name__ == "__main__":
    LABEL_COLS = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis",
                  "Crack", "Toothmark", "FurThick", "FurYellow"]

    img_dir = "C:/Users/msp/Tongue-AI-V2/images"
    fold_id = 1  # 測試第幾折

    val_csv = f"val_fold{fold_id}.csv"
    model_path = f"multibranch_tongue_fold{fold_id}.pth"

    validate_model(val_csv, img_dir, LABEL_COLS, model_path)
