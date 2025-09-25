import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_path, img_root, label_cols, train=True, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.label_cols = label_cols
        self.train = train
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size*1.15)),
                transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size*1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row['image_path'])
        full_path = os.path.join(self.img_root, path)
        if not os.path.exists(full_path):
            print(f'Image not found: {full_path}')
        img = Image.open(full_path).convert('RGB')
        x = self.tf(img)
        y = torch.tensor(row[self.label_cols].values.astype('float32'))  # shape: [8]
        return x, y
