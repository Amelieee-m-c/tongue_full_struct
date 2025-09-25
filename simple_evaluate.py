# simple_evaluate.py - HTTN 長尾驗證版
import torch
from torch.utils.data import DataLoader
from dataset import MultiLabelImageDataset
from model import Net
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, average_precision_score
import numpy as np
import sys

def evaluate_model(model_path, val_csv, image_root="images", batch_size=16):
    """驗證 HTTN 模型"""
    
    # 標籤定義
    labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    num_classes = len(labels)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    model = Net(num_classes=num_classes, backbone='resnet50', norm=True, scale=True)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    
    print(f"載入模型: {model_path}")
    print(f"驗證數據: {val_csv}")
    
    # 載入驗證集
    val_set = MultiLabelImageDataset(val_csv, image_root, labels, train=False, img_size=224)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"驗證樣本數: {len(val_set)}")
    
    # 儲存預測結果
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())
    
    P = torch.cat(all_probs).numpy()
    T = torch.cat(all_targets).numpy()
    
    # 自動計算每個類別最佳 threshold (long-tail)
    best_thr = []
    preds_thr = np.zeros_like(P)
    for i in range(num_classes):
        thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9
        best_f1 = 0.0
        best_t = 0.5
        for t in thresholds:
            pred_i = (P[:, i] >= t).astype(int)
            f1 = f1_score(T[:, i], pred_i)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thr.append(best_t)
        preds_thr[:, i] = (P[:, i] >= best_t).astype(int)
    
    # 整體指標
    micro_f1 = f1_score(T, preds_thr, average='micro')
    macro_f1 = f1_score(T, preds_thr, average='macro')
    mAP = average_precision_score(T, P, average='macro')
    
    print("\n整體指標")
    print("="*60)
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"mAP (Macro): {mAP:.4f}")
    
    # 各類別指標
    print("\n各類別指標")
    print("-"*60)
    print(f"{'Label':<15} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Jac':<6} {'Support':<8} {'Thr':<5}")
    print("-"*60)
    for i, label in enumerate(labels):
        f1 = f1_score(T[:, i], preds_thr[:, i])
        prec = precision_score(T[:, i], preds_thr[:, i])
        rec = recall_score(T[:, i], preds_thr[:, i])
        jac = jaccard_score(T[:, i], preds_thr[:, i])
        support = T[:, i].sum()
        thr = best_thr[i]
        print(f"{label:<15} {f1:<6.3f} {prec:<6.3f} {rec:<6.3f} {jac:<6.3f} {support:<8.0f} {thr:<5.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python simple_evaluate.py <模型路徑> <驗證CSV>")
        print("例如: python simple_evaluate.py model_best_img_httn.pth val_fold1.csv")
    else:
        evaluate_model(sys.argv[1], sys.argv[2])
