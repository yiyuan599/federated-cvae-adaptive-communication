# train_fashionmnist_teacher.py
# -*- coding: utf-8 -*-
"""
FashionMNIST 教师模型训练脚本
目标：准确率 > 92%（FashionMNIST 比 CIFAR-10 简单但比 MNIST 难）

模型设计：
- 输入：28×28×1 灰度图像
- CVAE 教师：~8M 参数
- CVAE 学生（small）：~1M 参数
- 潜在维度：128

使用方法:
  python train_fashionmnist_teacher.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# FashionMNIST CVAE 教师模型
# 输入：28×28×1 灰度图
# 编码器不使用标签，保证分类能力真实
# ============================================================
class FashionMNISTTeacherCVAE(nn.Module):
    def __init__(self, img_size=28, num_classes=10, latent_dim=128):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_channels = 1

        # 编码器: 28→14→7→3→1
        self.encoder = nn.Sequential(
            # Block 1: 28→14
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # Block 2: 14→7
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # Block 3: 7→3
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # Block 4: 3→1
            nn.Conv2d(512, 1024, 3, 1, 0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu     = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # 解码器输入
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 1024),
            nn.ReLU(),
        )

        # 解码器: 1→3→7→14→28
        self.decoder = nn.Sequential(
            # 1→3
            nn.ConvTranspose2d(1024, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 3→7
            nn.ConvTranspose2d(512, 256, 4, 2, 1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7→14
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 14→28
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, y):
        y_onehot = F.one_hot(y, self.num_classes).float()
        h = self.decoder_input(torch.cat([z, y_onehot], dim=1))
        h = h.view(h.size(0), 1024, 1, 1)
        return self.decoder(h)

    def classify(self, z):
        return self.classifier(z)

    def classify_from_image(self, x):
        mu, _ = self.encode(x)
        return self.classify(mu)

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar, z


# ============================================================
# FashionMNIST CVAE 学生模型（small）
# ============================================================
class FashionMNISTStudentCVAE(nn.Module):
    def __init__(self, img_size=28, num_classes=10, latent_dim=128,
                 student_size='small'):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_channels = 1

        ch = 32 if student_size == 'small' else 64

        # 编码器: 28→14→7→3→1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch*2, 4, 2, 1),
            nn.BatchNorm2d(ch*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1),
            nn.BatchNorm2d(ch*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch*4, ch*8, 3, 1, 0),
            nn.BatchNorm2d(ch*8),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu     = nn.Linear(ch*8, latent_dim)
        self.fc_logvar = nn.Linear(ch*8, latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + num_classes, ch*8),
            nn.ReLU(),
        )

        # 解码器: 1→3→7→14→28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch*8, ch*4, 3, 1, 0),
            nn.BatchNorm2d(ch*4),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1, output_padding=1),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.ConvTranspose2d(ch, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, y):
        y_onehot = F.one_hot(y, self.num_classes).float()
        ch8 = self.fc_mu.in_features  # ch*8
        h = self.decoder_input(torch.cat([z, y_onehot], dim=1))
        h = h.view(h.size(0), -1, 1, 1)
        return self.decoder(h)

    def classify(self, z):
        return self.classifier(z)

    def classify_from_image(self, x):
        mu, _ = self.encode(x)
        return self.classify(mu)

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar, z


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    return count_parameters(model) * 4 / 1024 / 1024


# ============================================================
# 数据加载
# ============================================================
def load_fashionmnist(data_dir='./data'):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)
    return train_ds, test_ds


# ============================================================
# 教师模型训练
# ============================================================
def train_teacher(quick=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    output_dir = './output/fashionmnist_teacher'
    os.makedirs(output_dir, exist_ok=True)

    train_ds, test_ds = load_fashionmnist()
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = FashionMNISTTeacherCVAE(latent_dim=128).to(device)
    print(f'教师模型参数量: {count_parameters(model):,} ({get_model_size_mb(model):.1f} MB)')

    num_epochs = 3 if quick else 60
    print(f'训练轮数: {num_epochs} {"(快速模式)" if quick else ""}')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    _mean = torch.tensor([0.2860], device=device).view(1, 1, 1, 1)
    _std  = torch.tensor([0.3530], device=device).view(1, 1, 1, 1)

    best_acc = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = cls_correct = total = 0

        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            recon, mu, logvar, z = model(x, y)
            x_01 = (x * _std + _mean).clamp(0, 1)
            recon_loss = F.mse_loss(recon, x_01)
            kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            logits     = model.classify(mu)
            cls_loss   = F.cross_entropy(logits, y)

            loss = recon_loss + 0.1 * kl_loss + 5.0 * cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            cls_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()

        # 评估
        model.eval()
        correct = val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model.classify_from_image(x)
                correct   += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)
        acc = correct / val_total
        train_acc = cls_correct / total

        print(f'Epoch {epoch:3d}: train_acc={train_acc*100:.2f}%  '
              f'test_acc={acc*100:.2f}%  loss={total_loss/len(train_loader):.4f}')

        history.append({'epoch': epoch, 'train_acc': train_acc, 'test_acc': acc})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{output_dir}/teacher_best.pt')
            print(f'  [保存] 最佳模型 acc={best_acc*100:.2f}%')

    torch.save(model.state_dict(), f'{output_dir}/teacher_final.pt')
    pd.DataFrame(history).to_csv(f'{output_dir}/teacher_history.csv', index=False)

    cfg = {
        'img_size': 28, 'num_classes': 10, 'latent_dim': 128,
        'best_acc': best_acc, 'num_epochs': num_epochs,
        'dataset': 'fashionmnist',
    }
    with open(f'{output_dir}/teacher_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f'\n教师模型训练完成！最佳准确率: {best_acc*100:.2f}%')
    print(f'模型保存至: {output_dir}/teacher_best.pt')
    return best_acc


if __name__ == '__main__':
    train_teacher()
