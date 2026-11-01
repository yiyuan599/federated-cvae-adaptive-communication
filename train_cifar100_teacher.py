# train_cifar100_teacher.py
# -*- coding: utf-8 -*-
"""
CIFAR-100 教师模型训练脚本
关键适配：num_classes=100, latent_dim=512, beta=0.01
目标准确率: Top-1 > 55%, Top-5 > 80%
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
from train_cifar10_teacher import (CIFAR10TeacherCVAE, CIFAR10StudentCVAE,
    count_parameters, get_model_size_mb)

# ── 超参配置 ───────────────────────────────────────
CONFIG = {
    'batch_size': 64,
    'lr': 3e-4,           # 100类需更小学习率
    'epochs': 80,         # 比CIFAR-10多训30轮
    'latent_dim': 512,    # 潜在空间加倍，100类条件生成需要
    'beta': 0.01,         # KL权重极小，防止100类条件生成时KL爆炸
    'cls_weight': 5.0,    # 提高分类损失权重
    'weight_decay': 1e-4,
}

def train_epoch_cifar100(model, loader, optimizer, device, config):
    model.train()
    total_loss = total_correct = total = 0
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    pbar = tqdm(loader, desc='Training', leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, z = model(data, labels)
        # CVAE 损失
        recon_loss = F.mse_loss(recon, data, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        cvae_loss = recon_loss + config['beta'] * kl_loss
        # 分类损失
        logits = model.classify(z)
        cls_loss = ce_loss_fn(logits, labels)
        loss = cvae_loss + config['cls_weight'] * cls_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'acc': f'{total_correct/total*100:.1f}%'})
    return {'loss': total_loss/len(loader), 'acc': total_correct/total}

def evaluate_top5(model, loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            logits = model.classify_from_image(data)
            # Top-1
            pred1 = logits.argmax(dim=1)
            correct1 += (pred1 == labels).sum().item()
            # Top-5
            _, top5 = logits.topk(5, dim=1)
            correct5 += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    return correct1/total, correct5/total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    # 数据增强（CIFAR-100 需要更强的增强）
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    train_ds = datasets.CIFAR100(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128,
                              shuffle=False, num_workers=2)
    # 模型：直接复用 CIFAR10TeacherCVAE，修改 num_classes=100, latent_dim=512
    teacher = CIFAR10TeacherCVAE(
        img_size=32, num_classes=100, latent_dim=CONFIG['latent_dim']
    ).to(device)
    student = CIFAR10StudentCVAE(
        img_size=32, num_classes=100, latent_dim=CONFIG['latent_dim'],
        student_size='small'
    ).to(device)
    print(f'教师参数: {count_parameters(teacher)/1e6:.2f}M')
    print(f'学生参数: {count_parameters(student)/1e6:.2f}M')
    optimizer = optim.AdamW(teacher.parameters(),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    # Cosine+Warmup 调度
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs']-warmup_epochs)
    output_dir = './output/cifar100_teacher'
    os.makedirs(output_dir, exist_ok=True)
    best_acc = 0
    history = []
    for epoch in range(CONFIG['epochs']):
        # Warmup
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = CONFIG['lr'] * (epoch+1) / warmup_epochs
        train_metrics = train_epoch_cifar100(
            teacher, train_loader, optimizer, device, CONFIG)
        top1, top5 = evaluate_top5(teacher, test_loader, device)
        if epoch >= warmup_epochs:
            scheduler.step()
        history.append({'epoch':epoch+1,'train_loss':train_metrics['loss'],
            'train_acc':train_metrics['acc'],'top1':top1,'top5':top5})
        print(f'Epoch {epoch+1}/{CONFIG["epochs"]}: '
              f'Loss={train_metrics["loss"]:.4f}, '
              f'Top1={top1*100:.2f}%, Top5={top5*100:.2f}%')
        if top1 > best_acc:
            best_acc = top1
            torch.save(teacher.state_dict(),
                       os.path.join(output_dir, 'teacher_best.pt'))
            print(f'  ** 新最佳 Top-1={best_acc*100:.2f}% **')
    torch.save(teacher.state_dict(), os.path.join(output_dir,'teacher_final.pt'))
    pd.DataFrame(history).to_csv(os.path.join(output_dir,'history.csv'),index=False)
    with open(os.path.join(output_dir,'config.json'),'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f'训练完成! 最佳Top-1={best_acc*100:.2f}%')
    print(f'模型保存: {output_dir}')

if __name__ == '__main__':
    main()
