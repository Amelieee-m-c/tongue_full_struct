import torch
import torch.nn as nn
from torchvision import models

def l2_norm(x, eps=1e-10):
    return x / (x.pow(2).sum(dim=1, keepdim=True).add_(eps).sqrt())

class ExtractorCNN(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.out_dim = 2048
            self.backbone = nn.Sequential(*(list(m.children())[:-1]))
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        return feat

class Embedding(nn.Module):
    def __init__(self, in_dim=2048, out_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)

class HTTNClassifier(nn.Module):
    def __init__(self, num_classes, in_dim=128, head_idx=None, tail_idx=None, use_bias=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=use_bias)
        self.head_idx = head_idx
        self.tail_idx = tail_idx
    def forward(self, x, h_feat=None, fuse_tail=True):
        logits = self.fc(x)
        if fuse_tail and (self.head_idx is not None) and (self.tail_idx is not None) and h_feat is not None:
            x_fused = x.clone()
            if x.size(0) == h_feat.size(0):
                mix_dim = int(x.size(1) * 0.5)
                x_fused[:, :mix_dim] = h_feat[:, :mix_dim]
                logits = self.fc(x_fused)
        return logits

class Net(nn.Module):
    def __init__(self, num_classes=36, norm=True, scale=True, backbone='resnet50', pretrained=True, head_idx=None, tail_idx=None):
        super(Net, self).__init__()
        self.extractor = ExtractorCNN(backbone=backbone, pretrained=pretrained)
        self.embedding = Embedding(in_dim=self.extractor.out_dim, out_dim=128)
        self.classifier = HTTNClassifier(num_classes, in_dim=128, head_idx=head_idx, tail_idx=tail_idx, use_bias=False)
        self.s = nn.Parameter(torch.tensor([10.0], dtype=torch.float32))
        self.norm = norm
        self.scale = scale
    def forward(self, x, h_feat=None, fuse_tail=False):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm: x = l2_norm(x)
        if self.scale: x = self.s * x
        logits = self.classifier(x, h_feat=h_feat, fuse_tail=fuse_tail)
        return logits
    @torch.no_grad()
    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = l2_norm(x)
        return x
    def weight_norm(self):
        with torch.no_grad():
            w = self.classifier.fc.weight.data
            self.classifier.fc.weight.data = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-12)
