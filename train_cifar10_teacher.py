# -*- coding: utf-8 -*-
"""
CIFAR-10 教师模型训练脚本
目标：
- 准确率 > 85%（CIFAR-10比MNIST难）
- 训练时间 ~1小时
- 模型大小 ~50MB（几十兆参数）

用于后续联邦学习和知识蒸馏实验
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CIFAR-10 CVAE 教师模型（~50M参数）
# 关键设计：编码器不使用标签，保证分类能力真实
# ============================================================
class CIFAR10TeacherCVAE(nn.Module):
    """
    针对CIFAR-10优化的大型教师CVAE模型
    - 输入：32x32x3 彩色图像
    - 参数量：~50M
    - 编码器：只看图像，不看标签
    - 解码器：使用标签作为条件
    """
    def __init__(self, img_size=32, num_classes=10, latent_dim=256):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_channels = 3
        
        # 编码器（更深更宽，~25M参数）
        # 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1
        self.encoder = nn.Sequential(
            # Block 1: 32 -> 16
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Block 2: 16 -> 8
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Block 3: 8 -> 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Block 4: 4 -> 2
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            # Block 5: 2 -> 1
            nn.Conv2d(1024, 2048, 2, 1, 0),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
        )
        
        # 潜在空间映射
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)
        
        # 解码器输入（潜在向量 + 条件标签）
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 2048),
            nn.ReLU(),
        )
        
        # 解码器（对称结构，~25M参数）
        self.decoder = nn.Sequential(
            # Block 1: 1 -> 2
            nn.ConvTranspose2d(2048, 1024, 2, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            # Block 2: 2 -> 4
            nn.ConvTranspose2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Block 3: 4 -> 8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Block 4: 8 -> 16
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 5: 16 -> 32
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        
        # 分类头（从潜在空间分类）
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    
    def encode(self, x):
        """编码：只输入图像，不使用标签"""
        batch_size = x.size(0)
        h = self.encoder(x)
        h = h.view(batch_size, -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """解码：从潜在变量和条件重构图像"""
        batch_size = z.size(0)
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        z_c = torch.cat([z, c_onehot], dim=1)
        h = self.decoder_input(z_c)
        h = h.view(batch_size, 2048, 1, 1)
        return self.decoder(h)
    
    def forward(self, x, c):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar, z
    
    def classify(self, z):
        """从潜在空间分类"""
        return self.classifier(z)
    
    def classify_from_image(self, x):
        """直接从图像分类（评估用）"""
        mu, _ = self.encode(x)
        return self.classifier(mu)


class CIFAR10StudentCVAE(nn.Module):
    """
    针对CIFAR-10的学生CVAE模型，支持多种规模
    通过 student_size 参数选择不同压缩程度

    规模说明（教师75.63M参数）：
        'large'  : ~25M参数, 压缩比~3x
        'medium' : ~12M参数, 压缩比~6x
        'small'  : ~6M参数,  压缩比~12x  (默认，与之前实验一致)
        'tiny'   : ~2M参数,  压缩比~30x
    """

    STUDENT_CONFIGS = {
        'large': {
            'channels': [96, 192, 384, 768],
            'use_double_conv': True,
            'cls_hidden': 384,
            'cls_dropout': 0.35,
        },
        'medium': {
            'channels': [96, 192, 384, 768],
            'use_double_conv': False,
            'cls_hidden': 384,
            'cls_dropout': 0.35,
        },
        'small': {
            'channels': [64, 128, 256, 512],
            'use_double_conv': False,
            'cls_hidden': 256,
            'cls_dropout': 0.3,
        },
        'tiny': {
            'channels': [32, 64, 128, 256],
            'use_double_conv': False,
            'cls_hidden': 128,
            'cls_dropout': 0.2,
        },
    }

    def __init__(self, img_size=32, num_classes=10, latent_dim=256, student_size='small'):
        super().__init__()
        assert student_size in self.STUDENT_CONFIGS, f"student_size must be one of {list(self.STUDENT_CONFIGS.keys())}"
        cfg = self.STUDENT_CONFIGS[student_size]

        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_channels = 3
        self.student_size = student_size

        ch = cfg['channels']
        top_ch = ch[-1]

        # --- 编码器 ---
        enc_layers = []
        if cfg['use_double_conv']:
            # large: 双卷积块提取更丰富特征
            in_c = 3
            for out_c in ch:
                enc_layers += [
                    nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2),
                    nn.Conv2d(out_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2),
                ]
                in_c = out_c
            # 2x2 -> 1x1
            enc_layers += [nn.Conv2d(top_ch, top_ch, 2, 1, 0), nn.BatchNorm2d(top_ch), nn.LeakyReLU(0.2)]
        else:
            in_c = 3
            for i, out_c in enumerate(ch):
                if i < len(ch) - 1:
                    enc_layers += [nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2)]
                else:
                    # 最后一层 4->1
                    enc_layers += [nn.Conv2d(in_c, out_c, 4, 1, 0), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2)]
                in_c = out_c

        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(top_ch, latent_dim)
        self.fc_logvar = nn.Linear(top_ch, latent_dim)

        # --- 解码器 ---
        self.decoder_input = nn.Sequential(nn.Linear(latent_dim + num_classes, top_ch), nn.ReLU())

        dec_layers = []
        rev_ch = list(reversed(ch))
        if cfg['use_double_conv']:
            # 1x1 -> 2x2
            dec_layers += [nn.ConvTranspose2d(top_ch, top_ch, 2, 1, 0), nn.BatchNorm2d(top_ch), nn.ReLU()]
            for i, out_c in enumerate(rev_ch[1:] + [3]):
                in_c = rev_ch[i]
                dec_layers += [
                    nn.ConvTranspose2d(in_c, in_c, 4, 2, 1), nn.BatchNorm2d(in_c), nn.ReLU(),
                    nn.Conv2d(in_c, out_c, 3, 1, 1),
                ]
                if out_c != 3:
                    dec_layers += [nn.BatchNorm2d(out_c), nn.ReLU()]
                else:
                    dec_layers.append(nn.Sigmoid())
        else:
            for i, out_c in enumerate(rev_ch[1:] + [3]):
                in_c = rev_ch[i]
                if i == 0:
                    dec_layers += [nn.ConvTranspose2d(in_c, out_c, 4, 1, 0)]
                else:
                    dec_layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)]
                if out_c != 3:
                    dec_layers += [nn.BatchNorm2d(out_c), nn.ReLU()]
                else:
                    dec_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec_layers)

        # --- 分类头 ---
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, cfg['cls_hidden']),
            nn.ReLU(),
            nn.Dropout(cfg['cls_dropout']),
            nn.Linear(cfg['cls_hidden'], num_classes),
        )

    def encode(self, x):
        batch_size = x.size(0)
        h = self.encoder(x)
        h = h.view(batch_size, -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        batch_size = z.size(0)
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        z_c = torch.cat([z, c_onehot], dim=1)
        h = self.decoder_input(z_c)
        h = h.view(batch_size, -1, 1, 1)
        return self.decoder(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar, z

    def classify(self, z):
        return self.classifier(z)

    def classify_from_image(self, x):
        mu, _ = self.encode(x)
        return self.classifier(mu)


# ============================================================
# 损失函数
# ============================================================
def cvae_loss_normalized(recon_x, x, mu, logvar, beta=1.0):
    """归一化的CVAE损失"""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ============================================================
# 训练函数
# ============================================================
def train_epoch(model, dataloader, optimizer, device, cls_weight=1.0, beta=0.1):
    """单个epoch训练"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_cls = 0
    correct = 0
    total = 0
    
    ce_loss_fn = nn.CrossEntropyLoss()
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        recon, mu, logvar, z = model(data, labels)
        
        # CVAE损失
        cvae_loss, recon_loss, kl_loss = cvae_loss_normalized(recon, data, mu, logvar, beta=beta)
        
        # 分类损失
        logits = model.classify(z)
        cls_loss = ce_loss_fn(logits, labels)
        
        # 总损失
        loss = cvae_loss + cls_weight * cls_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_cls += cls_loss.item()
        
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total*100:.1f}%'})
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
        'cls': total_cls / n_batches,
        'acc': correct / total
    }


def evaluate(model, dataloader, device):
    """评估模型分类准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            logits = model.classify_from_image(data)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """计算模型大小（MB）"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


# ============================================================
# 主训练脚本
# ============================================================
def main():
    print("=" * 60)
    print("  CIFAR-10 CVAE 教师模型训练")
    print("  目标：准确率 > 85%，模型 ~50MB")
    print("=" * 60)
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 超参数
    config = {
        'batch_size': 64,           # CIFAR需要较小的batch
        'lr': 5e-4,                 # 较小学习率
        'epochs': 50,               # 更多epoch
        'latent_dim': 256,          # 更大的潜在空间
        'beta': 0.05,               # 更小的KL权重
        'cls_weight': 3.0,          # 更高的分类权重
    }
    
    print(f"\n超参数配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 数据增强（CIFAR-10需要数据增强）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, 
                             shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\n数据集:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  类别: {train_dataset.classes}")
    
    # 模型
    teacher = CIFAR10TeacherCVAE(
        img_size=32,
        num_classes=10,
        latent_dim=config['latent_dim']
    ).to(device)
    
    student = CIFAR10StudentCVAE(
        img_size=32,
        num_classes=10,
        latent_dim=config['latent_dim']
    ).to(device)
    
    print(f"\n教师模型:")
    print(f"  参数量: {count_parameters(teacher):,} ({count_parameters(teacher)/1e6:.2f}M)")
    print(f"  模型大小: {get_model_size_mb(teacher):.2f} MB")
    
    print(f"\n学生模型:")
    print(f"  参数量: {count_parameters(student):,} ({count_parameters(student)/1e6:.2f}M)")
    print(f"  模型大小: {get_model_size_mb(student):.2f} MB")
    print(f"  压缩比: {count_parameters(teacher)/count_parameters(student):.2f}x")
    
    # 优化器
    optimizer = optim.AdamW(teacher.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # 训练历史
    history = {
        'epoch': [],
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'train_cls': [],
        'train_acc': [],
        'test_acc': [],
        'lr': [],
        'time': []
    }
    
    # 输出目录
    output_dir = './output/cifar10_teacher'
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_epoch(
            teacher, train_loader, optimizer, device,
            cls_weight=config['cls_weight'],
            beta=config['beta']
        )
        
        # 评估
        test_acc = evaluate(teacher, test_loader, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # 记录
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        history['train_cls'].append(train_metrics['cls'])
        history['train_acc'].append(train_metrics['acc'])
        history['test_acc'].append(test_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['time'].append(total_time)
        
        # 打印
        print(f"\nEpoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s, 总计: {total_time/60:.1f}min)")
        print(f"  Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon']:.4f}, "
              f"KL: {train_metrics['kl']:.4f}, Cls: {train_metrics['cls']:.4f})")
        print(f"  训练准确率: {train_metrics['acc']*100:.2f}%")
        print(f"  测试准确率: {test_acc*100:.2f}%")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(teacher.state_dict(), os.path.join(output_dir, 'teacher_best.pt'))
            print(f"  ** 新最佳模型已保存 (准确率: {best_acc*100:.2f}%) **")
        
        # 每10个epoch保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pt'))
    
    # 保存最终模型和历史
    torch.save(teacher.state_dict(), os.path.join(output_dir, 'teacher_final.pt'))
    torch.save(student.state_dict(), os.path.join(output_dir, 'student_init.pt'))
    
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总耗时: {(time.time()-start_time)/60:.1f} 分钟")
    print(f"最佳测试准确率: {best_acc*100:.2f}%")
    print(f"最终测试准确率: {history['test_acc'][-1]*100:.2f}%")
    print(f"\n模型已保存到: {output_dir}")
    
    return teacher, student, history


if __name__ == '__main__':
    main()

