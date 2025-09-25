import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm   # ✅ 新增

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
        body = img.crop((0, h//4, w, 3*h//4))   # 中間部分
        edge = img.crop((w//4, 0, 3*w//4, h))   # 左右裁掉
        whole = img                             # 原圖
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
# 3. 訓練流程 (含進度條)
# ------------------------
def train_multibranch(train_csv, val_csv, img_dir, label_cols, num_epochs=10, batch_size=8, lr=1e-4, fold_id=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = TongueDataset(train_csv, img_dir, label_cols, transform)
    val_ds = TongueDataset(val_csv, img_dir, label_cols, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiBranchTongueNet(num_classes=len(label_cols)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # ✅ tqdm 進度條
        pbar = tqdm(train_loader, desc=f"[Fold {fold_id}] Epoch {epoch+1}/{num_epochs}", unit="batch")
        for imgs, labels in pbar:
            imgs = [x.to(device) for x in imgs]
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})  # 顯示當前 batch loss

        avg_loss = total_loss / len(train_loader)
        print(f"[Fold {fold_id}] Epoch {epoch+1}/{num_epochs} 完成 ✅ Avg Loss: {avg_loss:.4f}")

    save_path = f"multibranch_tongue_fold{fold_id}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已存檔 {save_path}")

# ------------------------
# 4. 跑五折
# ------------------------
if __name__ == "__main__":
    LABEL_COLS = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis",
                  "Crack", "Toothmark", "FurThick", "FurYellow"]

    img_dir = "C:/Users/msp/Tongue-AI-V2/images"

    for fold in range(1, 6):
        train_csv = f"train_fold{fold}.csv"
        val_csv = f"val_fold{fold}.csv"
        train_multibranch(train_csv, val_csv,
                          img_dir=img_dir,
                          label_cols=LABEL_COLS,
                          num_epochs=20,
                          fold_id=fold)
