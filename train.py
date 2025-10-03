# confusion.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from train import MultiBranchTongueNet, TongueDataset   # ⚠️ 從 train.py 匯入
import pandas as pd

# 8 個類別
LABEL_COLS = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis",
              "Crack", "Toothmark", "FurThick", "FurYellow"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ 路徑
    weights = r"C:\Users\msp\Tongue-AI-V2\multibranch_tongue_fold1.pth"
    val_csv = r"C:\Users\msp\Tongue-AI-V2\train_fold1.csv"
    img_dir = r"C:\Users\msp\Tongue-AI-V2\images"

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset / Dataloader
    ds = TongueDataset(val_csv, img_dir, LABEL_COLS, transform)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    # Model
    model = MultiBranchTongueNet(num_classes=len(LABEL_COLS)).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = [x.to(device) for x in imgs]
            labels = labels.numpy()

            # 預測
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()

            # argmax → 單一類別
            preds = np.argmax(probs, axis=1)
            true_labels = np.argmax(labels, axis=1)

            all_preds.extend(preds)
            all_labels.extend(true_labels)

    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(LABEL_COLS))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_COLS,
                yticklabels=LABEL_COLS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("8x8 Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
