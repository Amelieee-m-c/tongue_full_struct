import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
from dataset import MultiLabelImageDataset
import numpy as np
from tqdm import tqdm     # ✅ 新增進度條

def make_loaders(train_csv, val_csv, img_root, label_cols,
                 batch_size=64, num_workers=4, img_size=224):
    train_set = MultiLabelImageDataset(train_csv, img_root, label_cols, train=True, img_size=img_size)
    val_set   = MultiLabelImageDataset(val_csv, img_root, label_cols, train=False, img_size=img_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def compute_pos_weight(train_loader, num_classes, device):
    counts_pos = torch.zeros(num_classes, device=device)
    counts_all = 0
    for _, y in tqdm(train_loader, desc="計算pos_weight", leave=False):
        y = y.to(device)
        counts_pos += y.sum(dim=0)
        counts_all += y.size(0)
    counts_neg = counts_all - counts_pos
    pos_weight = (counts_neg / (counts_pos.clamp(min=1.0))).float()
    return pos_weight

@torch.no_grad()
def get_head_feat(loader, net, head_idx, device):
    feats = []
    net.eval()
    for x, y in tqdm(loader, desc="蒐集head特徵", leave=False):
        x = x.to(device)
        y = y.to(device)
        mask = (y[:, head_idx].sum(dim=1) > 0)
        x_head = x[mask]
        if x_head.numel() > 0:
            feats.append(net.extract(x_head))
    return torch.cat(feats, dim=0) if feats else None

def train_one_epoch(model, loader, criterion, optimizer, device,
                    head_feat=None, head_idx=None, tail_idx=None, httn_prob=0.5):
    model.train()
    running = 0.0
    # ✅ tqdm 進度條
    pbar = tqdm(loader, desc="訓練中", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        fuse_tail = np.random.rand() < httn_prob
        logits = model(x, h_feat=head_feat, fuse_tail=fuse_tail)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device='cuda', thr=0.5):
    model.eval()
    from sklearn.metrics import f1_score, average_precision_score
    all_probs, all_targets = [], []
    pbar = tqdm(loader, desc="驗證中", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())
    P = torch.cat(all_probs).numpy()
    T = torch.cat(all_targets).numpy()
    micro_f1 = f1_score(T, (P >= thr).astype(np.int32), average='micro', zero_division=0)
    macro_f1 = f1_score(T, (P >= thr).astype(np.int32), average='macro', zero_division=0)
    mAP = average_precision_score(T, P, average='macro')
    return micro_f1, macro_f1, mAP

def main_img():
    label_cols = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis",
                  "Crack", "Toothmark", "FurThick", "FurYellow"]
    num_classes = len(label_cols)
    head_idx = list(range(6))
    tail_idx = list(range(6, 8))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    train_loader, val_loader = make_loaders(
        train_csv='train_fold1.csv',
        val_csv='val_fold1.csv',
        img_root='C:/Users/msp/Tongue-AI-V2/images',
        label_cols=label_cols,
        batch_size=64,
        num_workers=4,
        img_size=224
    )

    model = Net(num_classes=num_classes, norm=True, scale=True,
                backbone='resnet50', pretrained=True,
                head_idx=head_idx, tail_idx=tail_idx).to(device)

    pos_weight = compute_pos_weight(train_loader, num_classes, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    head_feat = get_head_feat(train_loader, model, head_idx, device)

    best_micro_f1 = 0.0
    for epoch in range(10):
        print(f"\n==== Epoch {epoch} ====")
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                               head_feat=head_feat, head_idx=head_idx,
                               tail_idx=tail_idx, httn_prob=0.5)
        micro, macro, mAP = evaluate(model, val_loader, device=device)
        print(f"[Epoch {epoch}] loss={loss:.4f} | microF1={micro:.4f} | macroF1={macro:.4f} | mAP={mAP:.4f}")
        if micro > best_micro_f1:
            best_micro_f1 = micro
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       'model_best_img_httn.pth')
            print("✅ 新最佳模型已儲存")

if __name__ == '__main__':
    main_img()
