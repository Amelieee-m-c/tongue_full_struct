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
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ---------- 1. Dataset ----------
class TongueDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.label_cols = ["TonguePale","TipSideRed","Spot","Ecchymosis",
                           "Crack","Toothmark","FurThick","FurYellow"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 多標籤轉單標籤（取第一個=1的index，全部為0則忽略）
        labels = row[self.label_cols].values.astype("int")
        if labels.sum() == 0:
            label = -1
        else:
            label = np.argmax(labels)
        return image, label

# ---------- 2. Transform ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------- 3. Dataset & DataLoader ----------
csv_file = "train_fold1.csv"
img_root = "C:/Users/msp/Tongue-AI-V2/images"
dataset = TongueDataset(csv_file, img_root, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# ---------- 4. Load Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 8)
model.load_state_dict(torch.load("multibranch_tongue_fold1.pth", map_location=device))
model.to(device)
model.eval()

# ---------- 5. Inference ----------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in dataloader:
        mask = labels != -1
        if mask.sum() == 0:
            continue
        images, labels = images[mask].to(device), labels[mask].to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------- 6. Confusion Matrix ----------
label_names = dataset.label_cols
cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("8-Class Confusion Matrix")
plt.show()

# ---------- 7. F1 分數 ----------
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Macro F1-score: {f1_macro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

# 可選：印出每個類別的 Precision / Recall / F1
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
