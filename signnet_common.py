# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# -------- 可按需調整的常用常數 --------
LABEL_COLS_DEFAULT = [
    "TonguePale", "TipSideRed", "Spot", "Ecchymosis",
    "Crack", "Toothmark", "FurThick", "FurYellow"
]

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- TIUO：旋正 / 對齊 / 方形化 -----------------
def tiuo_upright_image(img_pil: Image.Image, margin: float = 0.12, out_size: int | None = None) -> Image.Image:
    import numpy as _np
    img_np = _np.array(img_pil)
    gray = (0.299*img_np[...,0] + 0.587*img_np[...,1] + 0.114*img_np[...,2]).astype(_np.float32) if img_np.ndim==3 else img_np.astype(_np.float32)
    ys, xs = _np.where(gray > 1)
    if len(xs) < 20:
        return img_pil.resize((out_size,out_size)) if out_size else img_pil

    pts = _np.stack([xs, ys], 1).astype(_np.float32)
    mean = pts.mean(0, keepdims=True)
    cov = _np.cov((pts-mean).T)
    _, eigvecs = _np.linalg.eigh(cov)
    major = eigvecs[:, -1]
    ref = _np.array([0.0, 1.0], dtype=_np.float32)
    dot = (major*ref).sum()
    det = major[0]*ref[1] - major[1]*ref[0]
    angle_deg = float(_np.degrees(_np.arctan2(det, dot)))

    rotated = img_pil.rotate(-angle_deg, resample=Image.BILINEAR, expand=True, fillcolor=(0,0,0))
    rot_np = _np.array(rotated)
    rot_gray = (0.299*rot_np[...,0] + 0.587*rot_np[...,1] + 0.114*rot_np[...,2]).astype(_np.float32) if rot_np.ndim==3 else rot_np.astype(_np.float32)
    ys2, xs2 = _np.where(rot_gray > 1)
    if len(xs2)==0:
        return rotated.resize((out_size,out_size)) if out_size else rotated

    x1,x2 = xs2.min(), xs2.max()
    y1,y2 = ys2.min(), ys2.max()
    h,w = rot_gray.shape
    bw, bh = x2-x1+1, y2-y1+1
    pad_x, pad_y = int(round(margin*bw)), int(round(margin*bh))
    x1, y1 = max(0, x1-pad_x), max(0, y1-pad_y)
    x2, y2 = min(w-1, x2+pad_x), min(h-1, y2+pad_y)

    cropped = rotated.crop((x1,y1,x2+1,y2+1))
    cw,ch = cropped.size
    side = max(cw,ch)
    canvas = Image.new("RGB", (side,side), (0,0,0))
    canvas.paste(cropped, ((side-cw)//2, (side-ch)//2))
    return canvas.resize((out_size,out_size), resample=Image.BILINEAR) if out_size else canvas


# ----------------- Dataset：YOLO → SAM → TIUO -----------------
class TongueDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_cols=LABEL_COLS_DEFAULT,
                 transform=None,
                 yolo_model: YOLO|None=None,
                 sam_predictor: SamPredictor|None=None,
                 use_yolo=False, use_sam=False, use_tiuo=False,
                 augment=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_cols = label_cols
        self.use_yolo, self.use_sam, self.use_tiuo = use_yolo, use_sam, use_tiuo
        self.yolo_model, self.sam_predictor = yolo_model, sam_predictor
        # 基本變換
        base = [transforms.Resize((224,224)), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        # 簡單增強（訓練用）
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.85,1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1,0.1,0.1,0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.transform = transforms.Compose(base) if transform is None else transform

    def __len__(self): return len(self.data)

    @staticmethod
    def crop_parts(img: Image.Image):
        w,h = img.size
        body = img.crop((0, h//4, w, 3*h//4))
        edge = img.crop((w//4, 0, 3*w//4, h))
        whole = img
        return body, edge, whole

    def _yolo_crop(self, img: Image.Image) -> Image.Image:
        results = self.yolo_model.predict(img, verbose=False)
        if not (len(results) and len(results[0].boxes)): return img
        boxes = results[0].boxes
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        i = int(conf.argmax()); x1,y1,x2,y2 = map(int, xyxy[i])
        W,H = img.size
        x1,y1 = max(0,min(x1,W-1)), max(0,min(y1,H-1))
        x2,y2 = max(0,min(x2,W)),   max(0,min(y2,H))
        return img.crop((x1,y1,x2,y2)) if (x2>x1 and y2>y1) else img

    def _sam_mask(self, img: Image.Image) -> Image.Image:
        arr = np.array(img); self.sam_predictor.set_image(arr)
        h,w,_ = arr.shape
        masks,_,_ = self.sam_predictor.predict(
            point_coords=np.array([[w//2,h//2]]), point_labels=np.array([1]), multimask_output=False)
        mask = masks[0].astype(np.uint8)
        return Image.fromarray(arr * mask[...,None])

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["image_path"])).convert("RGB")
        if self.use_yolo and self.yolo_model is not None:  img = self._yolo_crop(img)
        if self.use_sam  and self.sam_predictor is not None: img = self._sam_mask(img)
        if self.use_tiuo: img = tiuo_upright_image(img)
        body, edge, whole = self.crop_parts(img)
        body, edge, whole = self.transform(body), self.transform(edge), self.transform(whole)
        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return (body, edge, whole), labels, row["image_path"]


# ----------------- MobileNetV3 backbone + 載入自定權重 -----------------
def _maybe_load_mnv3_state(backbone_features: nn.Module, state_path: str|None):
    if not state_path: return
    if not os.path.exists(state_path): 
        print(f"[warn] backbone init not found: {state_path}"); return
    try:
        sd = torch.load(state_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        missing, unexpected = backbone_features.load_state_dict(sd, strict=False)
        print(f"[backbone init] loaded with strict=False, missing={len(missing)}, unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[backbone init] failed: {e}")

class MobileNetV3Backbone(nn.Module):
    def __init__(self, variant="small", pretrained=True, out_dim=512, freeze_stages=0, init_state_path:str|None=None):
        super().__init__()
        if variant=="small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
        else:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.mobilenet_v3_large(weights=weights)

        self.features = base.features
        _maybe_load_mnv3_state(self.features, init_state_path)

        last_c = self.features[-1].out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(last_c, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Hardswish()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_stages>0:
            cut = min(len(self.features), {1:2, 2:4, 3:6, 4:8}.get(freeze_stages, 0))
            for p in self.features[:cut].parameters(): p.requires_grad=False

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        x = self.pool(x).flatten(1)
        return x


# ----------------- SignNet (MobileNetV3 版，多頭 + 兩個 Self-Attention) -----------------
class SignNetMobile(nn.Module):
    COLOR_IDX = [0,1,2,3]  # Pale, TipSideRed, Spot, Ecchymosis
    FUR_IDX   = [6,7]      # FurThick, FurYellow
    BODY_IDX  = [4]        # Crack
    EDGE_IDX  = [5]        # Toothmark

    def __init__(self, variant="small", feat_dim=512, attn_heads=4, dropout=0.1, freeze_stages=0, backbone_init_path:str|None=None):
        super().__init__()
        self.backbone_body  = MobileNetV3Backbone(variant, True, feat_dim, freeze_stages, backbone_init_path)
        self.backbone_edge  = MobileNetV3Backbone(variant, True, feat_dim, freeze_stages, backbone_init_path)
        self.backbone_whole = MobileNetV3Backbone(variant, True, feat_dim, freeze_stages, backbone_init_path)

        def proj(): return nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(True), nn.Dropout(dropout))
        self.proj_body, self.proj_edge, self.proj_whole = proj(), proj(), proj()

        self.attn_color = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=attn_heads, batch_first=True)
        self.attn_fur   = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=attn_heads, batch_first=True)

        self.head_body  = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.head_edge  = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.head_color = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, len(self.COLOR_IDX)))
        self.head_fur   = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, len(self.FUR_IDX)))

    def forward(self, inputs):
        body, edge, whole = inputs
        Fb = self.backbone_body(body);   Fb = self.proj_body(Fb)
        Fe = self.backbone_edge(edge);   Fe = self.proj_edge(Fe)
        Fw = self.backbone_whole(whole); Fw = self.proj_whole(Fw)

        # 個別 head（crack/toothmark）
        crack_logit     = self.head_body(Fb)
        toothmark_logit = self.head_edge(Fe)

        seq = torch.stack([Fb, Fw, Fe], dim=1)  # [N,3,C]
        color_ctx,_ = self.attn_color(seq, seq, seq); color_feat = color_ctx.mean(1); color_logits = self.head_color(color_feat)
        fur_ctx,_   = self.attn_fur(seq, seq, seq);   fur_feat   = fur_ctx.mean(1);   fur_logits   = self.head_fur(fur_feat)

        N = body.size(0); logits = torch.zeros(N, 8, device=body.device, dtype=color_logits.dtype)
        logits[:, self.COLOR_IDX] = color_logits
        logits[:, self.BODY_IDX]  = crack_logit
        logits[:, self.EDGE_IDX]  = toothmark_logit
        logits[:, self.FUR_IDX]   = fur_logits
        return logits


# ----------------- SAM 建立器 -----------------
def build_sam_predictor(variant: str, checkpoint: str, device: str):
    """
    Args:
        variant: "vit_h" | "vit_l" | "vit_b" | "mobile"
        checkpoint: 權重路徑（mobile_sam.pt 或官方 SAM 權重）
        device: "cuda" or "cpu"
    Returns:
        SamPredictor compatible 物件
    """
    ckpt_lower = checkpoint.lower()
    want_mobile = (variant == "mobile") or ("mobile_sam" in ckpt_lower) or ("mobilesam" in ckpt_lower)

    if want_mobile:
        # 1) 從 mobile_sam 套件載入（推薦）
        try:
            from mobile_sam import sam_model_registry as mobile_registry
            from mobile_sam import SamPredictor as MobileSamPredictor
            # 常見 key：vit_t
            for key in ["vit_t", "mobile", "vit_t_hq", "vit_t_mobile"]:
                if key in mobile_registry:
                    sam = mobile_registry[key](checkpoint=checkpoint)
                    sam.to(device)
                    return MobileSamPredictor(sam)
            # 如果上述 key 都不存在，丟錯
            raise KeyError(f"mobile_sam.sam_model_registry 沒有可用 key，現有: {list(mobile_registry.keys())}")
        except ImportError:
            # 沒裝或載入失敗，往下嘗試 segment_anything（某些 fork 也會註冊 vit_t/mobile）
            pass

    # 2) 官方 SAM 或嘗試 segment_anything 內的鍵
    from segment_anything import sam_model_registry as sa_registry
    from segment_anything import SamPredictor as SASamPredictor
    if want_mobile:
        # 嘗試在 segment_anything 內找 vit_t/mobile（某些 fork）
        for key in ["vit_t", "mobile"]:
            if key in sa_registry:
                sam = sa_registry[key](checkpoint=checkpoint)
                sam.to(device)
                return SASamPredictor(sam)
        raise RuntimeError(
            "偵測到你要用 Mobile-SAM，但找不到可用的 registry key。\n"
            "請先安裝 Mobile-SAM：pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
        )
    else:
        # 正統官方 SAM：vit_h / vit_l / vit_b
        if variant not in sa_registry:
            raise ValueError(f"SAM variant '{variant}' 不在 registry，可用: {list(sa_registry.keys())}")
        sam = sa_registry[variant](checkpoint=checkpoint)
        sam.to(device)
        return SASamPredictor(sam)

# ----------------- 簡單建立 YOLO/SAM（可在 train/validate 用） -----------------
def maybe_build_yolo(use_yolo: bool, weights: str|None, device: str):
    if not use_yolo: return None
    m = YOLO(weights); m.to(device); m.model.eval(); return m

def maybe_build_sam(use_sam: bool, variant: str, ckpt: str, device: str):
    if not use_sam:
        return None
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
    # 自動判斷 mobile_sam
    if (not variant) or (variant.strip() == ""):
        if ("mobile_sam" in ckpt.lower()) or ("mobilesam" in ckpt.lower()):
            variant = "mobile"
        else:
            variant = "vit_h"
    return build_sam_predictor(variant, ckpt, device)