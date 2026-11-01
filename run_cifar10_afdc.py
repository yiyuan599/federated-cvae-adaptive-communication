# run_cifar10_afdc.py
# -*- coding: utf-8 -*-
"""
AFDC 在 CIFAR-10 上的完整实验脚本
复用 train_cifar10_teacher.py 中的模型类，适配 CIFAR-10 10 类场景

由 run_cifar100_afdc.py 改写而来，已包含调试过程中确认的全部修复：
  1. 潜在分布统计使用客户端本地编码器 encode(x)，并采用确定性变换，
     使漂移反映本地训练后的真实分布变化（而非随机数据增强噪声）
  2. 聚合按每个客户端在该类别上的样本数加权（而非客户端总样本数）
  3. 单样本类别的标准差改用 unbiased=False，避免 NaN
  4. 本地训练 DataLoader 加 drop_last=True，避免 batch_size=1 时 BatchNorm 报错
  5. 统计量收集向量化（整批 .cpu() 搬运，避免逐样本同步）
  6. 本地训练 batch_size 64→256
  7. 通信调度：CVAE 潜在分布 KL 漂移触发稀疏上传（comm_mode='sparse'）
  8. 自适应通信阈值：comm_mode='adaptive' 时阈值随全局漂移 EMA 动态缩放

使用方法:
  # 实验A：基础横向对比
  python run_cifar10_afdc.py --exp baseline --partition dirichlet --alpha 0.5

  # 实验B：异质性纵向扫描
  python run_cifar10_afdc.py --exp alpha_scan

  # 实验C：压缩比扫描
  python run_cifar10_afdc.py --exp compression

  # 实验E：客户端规模扫描
  python run_cifar10_afdc.py --exp scale_scan

  # 实验F：消融实验
  python run_cifar10_afdc.py --exp ablation

  # 快速测试
  python run_cifar10_afdc.py --exp baseline --quick
"""

import os, sys, json, time, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cifar10_teacher import (CIFAR10TeacherCVAE, CIFAR10StudentCVAE,
    count_parameters, get_model_size_mb)
from datasets.data_partition import FederatedDataPartitioner

LATENT_DIM = 256
NUM_CLASSES = 10
NUM_REPEATS = 3
RANDOM_SEEDS = [42, 123, 999]


# ============================================================
# 工具函数
# ============================================================
def find_teacher(base='./output/cifar10_teacher'):
    path = os.path.join(base, 'teacher_best.pt')
    return path if os.path.exists(path) else None


def load_cifar10(data_dir='./data'):
    # 教师模型训练时在原始 [0,1] 像素域，不做 Normalize，此处保持一致
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return train_ds, test_ds


def evaluate_topk(model, loader, device, k=5):
    model.eval()
    c1 = c5 = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model.classify_from_image(x)
            c1 += (logits.argmax(1) == y).sum().item()
            _, topk = logits.topk(min(k, NUM_CLASSES), dim=1)
            c5 += topk.eq(y.unsqueeze(1)).any(1).sum().item()
            total += y.size(0)
    return c1 / total, c5 / total


def partition_data(train_ds, partition_type, num_clients, alpha=0.5, seed=42):
    """数据分区，返回客户端 Subset 列表"""
    partitioner = FederatedDataPartitioner(train_ds, num_clients, seed=seed)
    if partition_type == 'iid':
        partition = partitioner.partition_iid()
    elif partition_type == 'dirichlet':
        partition = partitioner.partition_dirichlet(alpha=alpha)
    elif partition_type == 'label_skew':
        partition = partitioner.partition_label_skew(labels_per_client=2)
    else:
        raise ValueError(f"未知分区类型: {partition_type}")
    return partitioner.get_client_datasets(partition)


# ============================================================
# CVAE 本地训练辅助
# ============================================================
def train_local_cvae(model, dataset, device, epochs=5, lr=5e-4, beta=0.01,
                     cls_weight=5.0):
    """客户端本地 CVAE 训练"""
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            recon, mu, logvar, z = model(x, y)
            recon_l = F.mse_loss(recon, x, reduction='mean')
            kl_l = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            logits = model.classify(z)
            cls_l = F.cross_entropy(logits, y)
            loss = recon_l + beta * kl_l + cls_weight * cls_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()


# ============================================================
# AFDC 虚拟数据生成 (从聚合的潜在分布参数)
# ============================================================
@torch.no_grad()
def generate_virtual_data_from_stats(decoder_model, mu_global, sigma_global,
                                      device, num_per_class=100):
    """
    从全局聚合的潜在分布参数 (mu, sigma) 生成虚拟数据。
    AFDC 的核心：不上传解码器权重，只上传潜在分布统计。
    """
    decoder_model.eval()
    all_images, all_labels = [], []

    for cls in range(NUM_CLASSES):
        if mu_global is not None and cls in mu_global:
            mu_c = mu_global[cls].to(device)
            sigma_c = sigma_global[cls].to(device)
            eps = torch.randn(num_per_class, LATENT_DIM).to(device)
            z = mu_c.unsqueeze(0) + sigma_c.unsqueeze(0) * eps
        else:
            z = torch.randn(num_per_class, LATENT_DIM).to(device)

        lbl = torch.full((num_per_class,), cls, dtype=torch.long).to(device)
        imgs = decoder_model.decode(z, lbl).clamp(0, 1)
        all_images.append(imgs.cpu())
        all_labels.append(lbl.cpu())

    return torch.cat(all_images), torch.cat(all_labels)


# ============================================================
# AFDC 核心：多轮联邦蒸馏
# ============================================================
def run_afdc(teacher, train_ds, test_loader, device,
             partition_type='dirichlet', alpha=0.5, num_clients=10,
             student_size='small', num_rounds=20, local_epochs=5,
             T=4.0,
             w_latent=0.3, w_recon=0.2, w_cvae=0.1,
             use_adaptive=True, T_check=5,
             comm_mode='full', comm_tau=0.05, min_upload=3,
             comm_beta=0.8, comm_tau_min=0.01, comm_tau_max=1.0,
             seed=42, output_dir='./output/tmp'):
    """
    AFDC 完整多轮联邦蒸馏流程:

    每轮:
    1. 各客户端本地训练 CVAE → 提取类别级 (mu, sigma) 潜在分布参数
    2. 服务器按数据量加权聚合 → 全局混合高斯先验
    3. 服务器从全局先验采样 → 生成覆盖全部类别的虚拟数据
    4. 教师模型对虚拟数据推理软标签
    5. 各客户端接收虚拟数据+软标签 → 三目标蒸馏 (KD+CE+Align)
    6. (可选) KL 散度检测漂移 → 自适应调度权重
    7. (可选) 漂移触发式稀疏通信：仅漂移超过阈值的客户端上传潜在统计，
       服务器复用上次上传的统计参与聚合；min_upload 保证最低参与度
    8. (可选) 自适应阈值：阈值随全局漂移尺度动态调整（EMA），
       漂移大时提高阈值、漂移小时降低阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 统计提取用确定性测试变换，避免随机增强被误判为潜在分布漂移
    stats_transform = getattr(test_loader.dataset, 'transform', None)

    # 数据分区
    client_datasets = partition_data(
        train_ds, partition_type, num_clients, alpha, seed)

    # 创建学生模型
    student = CIFAR10StudentCVAE(
        img_size=32, num_classes=NUM_CLASSES,
        latent_dim=LATENT_DIM, student_size=student_size
    ).to(device)

    # 全局模型（教师）作为中央解码器
    global_model = copy.deepcopy(teacher).to(device)
    global_model.eval()

    # 为每个客户端创建本地 CVAE 副本
    client_models = []
    for cid in range(num_clients):
        cm = copy.deepcopy(teacher).to(device)
        client_models.append(cm)

    # 历史记录
    history = []
    # 自适应调度状态
    client_drift_history = {cid: [] for cid in range(num_clients)}
    prev_mu_stats = {cid: {} for cid in range(num_clients)}
    prev_sigma_stats = {}
    # 通信调度状态：服务器缓存每个客户端上次上传的潜在分布统计
    uploaded_mu_stats = {cid: None for cid in range(num_clients)}
    uploaded_sigma_stats = {cid: None for cid in range(num_clients)}
    uploaded_count_stats = {cid: None for cid in range(num_clients)}
    bytes_per_client = NUM_CLASSES * LATENT_DIM * 2 * 4 + NUM_CLASSES * 4
    cum_comm_bytes = 0
    drift_ema = None
    # 每轮每客户端的漂移明细（供论文画图：round_drift_scores 落盘）
    drift_records = []

    print(f'AFDC 配置: {partition_type}, alpha={alpha}, clients={num_clients}, '
          f'student={student_size}, adaptive={use_adaptive}')
    print(f'  蒸馏权重: w_latent={w_latent}, w_recon={w_recon}, w_cvae={w_cvae}')

    # 学生模型的优化器与学习率调度只创建一次（不能每轮重建，否则动量清零、
    # 学习率全程恒定不衰减，越到后面越容易把已经学好的参数冲飞——即"早衰"的根因）
    opt_s = optim.Adam(student.parameters(), lr=5e-4)
    scheduler_s = optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=num_rounds)

    best_top1 = 0
    best_top5 = 0

    for round_idx in range(num_rounds):
        print(f'\n--- AFDC Round {round_idx+1}/{num_rounds} ---')

        # ── Step 1-2: 各客户端本地训练 CVAE → 上传潜在分布参数 ──
        all_mu_stats = []
        all_sigma_stats = []
        all_class_counts = []
        round_drift_scores = []

        for cid in range(num_clients):
            cds = client_datasets[cid]
            class_zs = {c: [] for c in range(NUM_CLASSES)}
            if len(cds) > 0:
                # 本地训练
                train_local_cvae(client_models[cid], cds, device,
                                 epochs=local_epochs, lr=5e-4, beta=0.01, cls_weight=5.0)

                # 提取每个类别的潜在分布统计 (mu, sigma)
                client_models[cid].eval()
                c_loader = DataLoader(cds, batch_size=256, shuffle=False, num_workers=0)

                # 按类别累积潜在变量（整批搬运，向量化分桶，避免逐样本 .cpu() 同步）
                orig_transform = train_ds.transform
                if stats_transform is not None:
                    train_ds.transform = stats_transform
                try:
                    with torch.no_grad():
                        for x, y in c_loader:
                            x = x.to(device)
                            mu, _ = client_models[cid].encode(x)
                            mu_cpu = mu.cpu()
                            for c in range(NUM_CLASSES):
                                mask = (y == c)
                                if mask.any():
                                    class_zs[c].append(mu_cpu[mask])
                finally:
                    train_ds.transform = orig_transform

            # 每个类别的 mu, sigma
            mu_stats = {}
            sigma_stats = {}
            count_stats = {}
            for c in range(NUM_CLASSES):
                if class_zs[c]:
                    zs = torch.cat(class_zs[c], dim=0)
                    mu_stats[c] = zs.mean(dim=0)
                    sigma_stats[c] = zs.std(dim=0, unbiased=False).clamp(min=0.01)
                    count_stats[c] = zs.shape[0]
                else:
                    mu_stats[c] = torch.zeros(LATENT_DIM)
                    sigma_stats[c] = torch.ones(LATENT_DIM)
                    count_stats[c] = 0

            all_mu_stats.append(mu_stats)
            all_sigma_stats.append(sigma_stats)
            all_class_counts.append(count_stats)

            # 通信调度：计算本轮统计与上次上传统计的 KL 漂移
            if (comm_mode in ('sparse', 'adaptive') and round_idx > 0
                    and uploaded_mu_stats[cid] is not None):
                round_drift_scores.append((
                    cid,
                    compute_drift(mu_stats, sigma_stats,
                                  uploaded_mu_stats[cid],
                                  uploaded_sigma_stats[cid])
                ))

            # 自适应调度：计算KL散度漂移
            if use_adaptive and round_idx > 0 and round_idx % T_check == 0:
                drift = compute_drift(
                    mu_stats, sigma_stats, prev_mu_stats[cid],
                    prev_sigma_stats.get(cid, {}))
                client_drift_history[cid].append(drift)

        # ── 通信调度决策：漂移触发式稀疏上传 ──
        if comm_mode == 'sparse' and round_idx > 0:
            round_drift_scores.sort(key=lambda x: x[1], reverse=True)
            upload_set = {cid for cid, d in round_drift_scores if d >= comm_tau}
            effective_tau = comm_tau
            # 保底：至少 min_upload 个客户端上传，避免全局统计长期静止
            for cid, d in round_drift_scores:
                if len(upload_set) >= min_upload:
                    break
                upload_set.add(cid)
        elif comm_mode == 'adaptive' and round_idx > 0:
            drifts = [d for _, d in round_drift_scores]
            med = float(np.median(drifts)) if drifts else 0.0
            drift_ema = med if drift_ema is None else 0.8 * drift_ema + 0.2 * med
            effective_tau = float(np.clip(comm_beta * drift_ema,
                                          comm_tau_min, comm_tau_max))
            round_drift_scores.sort(key=lambda x: x[1], reverse=True)
            upload_set = {cid for cid, d in round_drift_scores if d >= effective_tau}
            for cid, d in round_drift_scores:
                if len(upload_set) >= min_upload:
                    break
                upload_set.add(cid)
        else:
            upload_set = set(range(num_clients))
            effective_tau = 0.0

        # 更新服务器缓存并统计本轮通信量
        round_comm_bytes = 0
        for cid in range(num_clients):
            if cid in upload_set:
                uploaded_mu_stats[cid] = all_mu_stats[cid]
                uploaded_sigma_stats[cid] = all_sigma_stats[cid]
                uploaded_count_stats[cid] = all_class_counts[cid]
                round_comm_bytes += bytes_per_client
        cum_comm_bytes += round_comm_bytes

        # 记录本轮各客户端相对上次上传版本的 KL 漂移
        for cid, d in round_drift_scores:
            drift_records.append({
                'round': round_idx + 1,
                'client_id': cid,
                'drift': float(d),
            })
        round_drifts = [d for _, d in round_drift_scores]

        # 保存本轮统计用于下一轮漂移检测
        if use_adaptive:
            prev_mu_stats = {cid: all_mu_stats[cid] for cid in range(num_clients)}
            prev_sigma_stats = {cid: all_sigma_stats[cid] for cid in range(num_clients)}

        # ── Step 3-4: 服务器聚合 → 生成虚拟数据 → 教师推理软标签 ──
        # 聚合潜在分布：按每个客户端在该类别上的样本数加权（而非客户端总样本数），
        # 否则某类样本极少的客户端会因其它类别样本多而在该类别上获得不成比例的权重
        mu_global = {}
        sigma_global = {}
        for c in range(NUM_CLASSES):
            total_c = sum(uploaded_count_stats[cid][c] for cid in range(num_clients))
            if total_c > 0:
                mu_c = sum(uploaded_mu_stats[cid][c] * uploaded_count_stats[cid][c] / total_c
                          for cid in range(num_clients))
                sigma_c = sum(uploaded_sigma_stats[cid][c] * uploaded_count_stats[cid][c] / total_c
                             for cid in range(num_clients))
            else:
                mu_c = torch.zeros(LATENT_DIM)
                sigma_c = torch.ones(LATENT_DIM)
            mu_global[c] = mu_c
            sigma_global[c] = sigma_c.clamp(min=0.01)

        # 从全局先验生成虚拟数据
        virtual_images, virtual_labels = generate_virtual_data_from_stats(
            global_model, mu_global, sigma_global, device,
            num_per_class=100)

        # 教师对虚拟数据推理软标签
        virtual_ds = torch.utils.data.TensorDataset(virtual_images, virtual_labels)
        virtual_loader = DataLoader(virtual_ds, batch_size=64, shuffle=True, drop_last=True)
        print(f'  虚拟数据: {len(virtual_ds)} 样本')

        # ── Step 4: 知识蒸馏 (使用虚拟数据) ──
        student.train()
        teacher.eval()

        for epoch in range(2):  # 每轮蒸馏 2 epoch
            for x, y in virtual_loader:
                x, y = x.to(device), y.to(device)
                opt_s.zero_grad()

                # 学生前向
                s_recon, s_mu, s_logvar, s_z = student(x, y)
                s_logits = student.classify(s_z)

                # 教师前向
                with torch.no_grad():
                    t_recon, t_mu, t_logvar, t_z = teacher(x, y)
                    t_logits = teacher.classify(t_z)

                # ── 计算三目标蒸馏损失 ──
                soft_loss = F.kl_div(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T ** 2)

                hard_loss = F.cross_entropy(s_logits, y)

                s_feat = F.normalize(s_mu, dim=1)
                t_feat = F.normalize(t_mu, dim=1)
                align_loss = 1.0 - (s_feat * t_feat).sum(dim=1).mean()

                student_cvae = F.mse_loss(s_recon, x) + \
                    0.01 * (-0.5 * torch.mean(1 + s_logvar - s_mu.pow(2) - s_logvar.exp()))

                # 线性渐进权重调度（避免阶跃跳变导致的性能崩溃）
                progress = round_idx / max(1, num_rounds - 1)
                alpha_kd = 0.3 + 0.5 * progress   # 0.3 → 0.8
                alpha_ce = 1.0 - alpha_kd

                # 自适应调整
                if use_adaptive and round_idx > 0:
                    if client_drift_history:
                        all_drifts = []
                        for cid in range(num_clients):
                            all_drifts.extend(client_drift_history[cid][-1:] if client_drift_history[cid] else [0])
                        avg_drift = np.mean(all_drifts) if all_drifts else 0
                        if avg_drift > 0.5:
                            alpha_kd += 0.1

                total_loss = (
                    alpha_kd * soft_loss +
                    alpha_ce * hard_loss +
                    w_latent * align_loss +
                    w_cvae * student_cvae
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_s.step()

        scheduler_s.step()

        # ── 评估本轮 ──
        top1, top5 = evaluate_topk(student, test_loader, device)

        if top1 > best_top1:
            best_top1 = top1
            best_top5 = top5

        history.append({
            'round': round_idx + 1,
            'top1': top1,
            'top5': top5,
            'best_top1': best_top1,
            'uploaded_clients': len(upload_set),
            'upload_ratio': len(upload_set) / num_clients,
            'comm_bytes': round_comm_bytes,
            'cum_comm_bytes': cum_comm_bytes,
            'adaptive_tau': effective_tau,
            'drift_median': float(np.median(round_drifts)) if round_drifts else 0.0,
            'drift_min': float(np.min(round_drifts)) if round_drifts else 0.0,
            'drift_max': float(np.max(round_drifts)) if round_drifts else 0.0,
        })
        print(f'  Round {round_idx+1}: Top1={top1*100:.2f}%, Top5={top5*100:.2f}% '
              f'(best={best_top1*100:.2f}%)')

    # 保存结果
    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, 'afdc_history.csv'), index=False)
    if drift_records:
        pd.DataFrame(drift_records).to_csv(
            os.path.join(output_dir, 'drift_history.csv'), index=False)
    torch.save(student.state_dict(),
               os.path.join(output_dir, 'afdc_student_final.pt'))

    return {
        'method': 'AFDC',
        'top1': history[-1]['top1'],
        'top5': history[-1]['top5'],
        'best_top1': best_top1,
        'best_top5': best_top5,
        'partition': partition_type,
        'student_size': student_size,
        'num_clients': num_clients,
        'upload_ratio_mean': np.mean([h['upload_ratio'] for h in history]),
        'comm_bytes_total': cum_comm_bytes,
        'comm_bytes_baseline': num_rounds * num_clients * bytes_per_client,
        'comm_saving': 1.0 - cum_comm_bytes / (
            num_rounds * num_clients * bytes_per_client),
        'history': history,
    }


def compute_drift(mu_stats, sigma_stats, prev_mu, prev_sigma):
    """计算 KL 散度漂移量"""
    drifts = []
    for c in range(NUM_CLASSES):
        if c in mu_stats and c in prev_mu:
            mu_cur = mu_stats[c]
            mu_prev = prev_mu[c]
            sigma_cur = sigma_stats.get(c, torch.ones(LATENT_DIM) * 0.1)
            sigma_prev = prev_sigma.get(c, torch.ones(LATENT_DIM) * 0.1)

            var_cur = sigma_cur ** 2
            var_prev = sigma_prev ** 2
            kl = 0.5 * (
                ((mu_cur - mu_prev) ** 2) / var_prev.clamp(min=1e-6) +
                var_cur / var_prev.clamp(min=1e-6) -
                1.0 +
                torch.log(var_prev.clamp(min=1e-6) / var_cur.clamp(min=1e-6))
            ).mean().item()
            drifts.append(max(0, kl))
    return np.mean(drifts) if drifts else 0.0


# ============================================================
# 重复运行
# ============================================================
def run_with_repeats(run_fn, *args, num_repeats=NUM_REPEATS, **kwargs):
    results_list = []
    base_out = kwargs.pop('output_dir', './output/tmp')
    for i in range(num_repeats):
        seed = RANDOM_SEEDS[i]
        out_dir = f'{base_out}_rep{i}'
        print(f'\n--- 第 {i+1}/{num_repeats} 次 (seed={seed}) ---')
        try:
            r = run_fn(*args, **kwargs, seed=seed, output_dir=out_dir)
            results_list.append(r)
            print(f'  结果: Top1={r["top1"]*100:.2f}%')
        except Exception as e:
            print(f'  [警告] 第{i+1}次运行失败: {e}')
            import traceback
            traceback.print_exc()
            results_list.append(None)

    valid = [r for r in results_list if r is not None]
    if not valid:
        raise RuntimeError("所有重复运行均失败!")

    top1s = [r['top1'] for r in valid]
    top5s = [r['top5'] for r in valid]
    best_top1s = [r.get('best_top1', r['top1']) for r in valid]

    summary = {}
    for k in ['method', 'partition', 'student_size']:
        if k in valid[0]:
            summary[k] = valid[0][k]
    summary.update({
        'top1_mean': np.mean(top1s),
        'top1_std':  np.std(top1s),
        'top5_mean': np.mean(top5s),
        'top5_std':  np.std(top5s),
        'best_top1_mean': np.mean(best_top1s),
        'best_top1_std':  np.std(best_top1s),
        'top1_individual': top1s,
        'top5_individual': top5s,
        'num_valid': len(valid),
    })
    for k in ['upload_ratio_mean', 'comm_bytes_total',
              'comm_bytes_baseline', 'comm_saving']:
        vals = [r.get(k) for r in valid]
        if all(v is not None for v in vals):
            summary[k] = float(np.mean(vals))
    return summary


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='AFDC CIFAR-10 实验')
    parser.add_argument('--exp', default='baseline',
                        choices=['baseline', 'alpha_scan', 'compression',
                                 'scale_scan', 'ablation', 'comm_adaptive'])
    parser.add_argument('--partition', default='dirichlet',
                        choices=['iid', 'dirichlet', 'label_skew'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--student_size', default='small',
                        choices=['large', 'medium', 'small', 'tiny'])
    parser.add_argument('--min_upload', type=int, default=3,
                        help='稀疏通信保底上传客户端数')
    parser.add_argument('--comm_taus', default=None,
                        help='稀疏通信阈值列表，逗号分隔，如 0.3,0.5')
    parser.add_argument('--comm_beta', type=float, default=0.8,
                        help='自适应阈值缩放系数 (adaptive 模式)')
    parser.add_argument('--comm_tau_max', type=float, default=1.0,
                        help='自适应阈值上限')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试: 减少轮次/epoch, 不重复')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # 加载教师模型
    teacher_path = find_teacher()
    if not teacher_path:
        print('[错误] 未找到CIFAR-10教师模型，请先运行 train_cifar10_teacher.py')
        return

    teacher = CIFAR10TeacherCVAE(
        img_size=32, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device,
                                        weights_only=True))
    teacher.eval()
    print(f'教师模型已加载: {teacher_path}')

    train_ds, test_ds = load_cifar10()
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    repeats = 1 if args.quick else NUM_REPEATS
    rounds = 5 if args.quick else 20
    local_ep = 2 if args.quick else 5

    os.makedirs('./output', exist_ok=True)

    # ── 实验A: 基础横向对比 ──
    if args.exp == 'baseline':
        results = []
        for pt in ['iid', 'dirichlet', 'label_skew']:
            print(f'\n{"="*50}\n  AFDC | {pt}\n{"="*50}')
            alpha_val = args.alpha if pt == 'dirichlet' else 0.5
            out_dir = f'./output/cifar10_afdc_baseline_{pt}'
            r = run_with_repeats(
                run_afdc, teacher, train_ds, test_loader, device,
                partition_type=pt, alpha=alpha_val,
                num_clients=args.num_clients,
                student_size=args.student_size,
                num_rounds=rounds, local_epochs=local_ep,
                num_repeats=repeats,
                output_dir=out_dir,
            )
            r['partition'] = pt
            r['alpha'] = alpha_val
            results.append(r)
            print(f'  --> {pt}: Top1={r["top1_mean"]*100:.2f}% '
                  f'± {r["top1_std"]*100:.2f}%')

        df = pd.DataFrame(results)
        df.to_csv('./output/cifar10_afdc_baseline.csv', index=False)
        print(f'\n实验A完成! 结果: ./output/cifar10_afdc_baseline.csv')

    # ── 实验G: 通信调度对照（漂移触发式稀疏上传） ──
    elif args.exp == 'comm_adaptive':
        if args.comm_taus:
            tau_values = [float(x) for x in args.comm_taus.split(',')]
        else:
            tau_values = [0.05, 0.1, 0.2] if not args.quick else [0.1]
        alpha_val = args.alpha if args.partition == 'dirichlet' else 0.5
        results = []

        # baseline: 每轮全量上传
        print(f'\n{"="*50}\n  Comm | full | {args.partition}\n{"="*50}')
        out_dir = f'./output/cifar10_comm_full_{args.partition}'
        r = run_with_repeats(
            run_afdc, teacher, train_ds, test_loader, device,
            partition_type=args.partition, alpha=alpha_val,
            num_clients=args.num_clients,
            student_size=args.student_size,
            num_rounds=rounds, local_epochs=local_ep,
            comm_mode='full',
            num_repeats=repeats,
            output_dir=out_dir,
        )
        r['comm_mode'] = 'full'
        r['comm_tau'] = 0.0
        r['min_upload'] = args.min_upload
        r['partition'] = args.partition
        r['alpha'] = alpha_val
        results.append(r)
        print(f'  --> full: Top1={r["top1_mean"]*100:.2f}% '
              f'comm_saving={r.get("comm_saving", 0.0)*100:.1f}%')

        # 方案A: 漂移触发式稀疏上传（阈值扫描）
        for tau in tau_values:
            print(f'\n{"="*50}\n  Comm | sparse tau={tau} | {args.partition}\n{"="*50}')
            out_dir = (f'./output/cifar10_comm_sparse_tau{tau}_'
                       f'mu{args.min_upload}_{args.partition}')
            r = run_with_repeats(
                run_afdc, teacher, train_ds, test_loader, device,
                partition_type=args.partition, alpha=alpha_val,
                num_clients=args.num_clients,
                student_size=args.student_size,
                num_rounds=rounds, local_epochs=local_ep,
                comm_mode='sparse', comm_tau=tau,
                min_upload=args.min_upload,
                num_repeats=repeats,
                output_dir=out_dir,
            )
            r['comm_mode'] = 'sparse'
            r['comm_tau'] = tau
            r['min_upload'] = args.min_upload
            r['partition'] = args.partition
            r['alpha'] = alpha_val
            results.append(r)
            print(f'  --> sparse tau={tau}: Top1={r["top1_mean"]*100:.2f}% '
                  f'upload_ratio={r.get("upload_ratio_mean", 1.0)*100:.1f}% '
                  f'comm_saving={r.get("comm_saving", 0.0)*100:.1f}%')

        # 自适应阈值调度：阈值随全局漂移 EMA 动态缩放
        print(f'\n{"="*50}\n  Comm | adaptive | {args.partition}\n{"="*50}')
        out_dir = f'./output/cifar10_comm_adaptive_{args.partition}'
        r = run_with_repeats(
            run_afdc, teacher, train_ds, test_loader, device,
            partition_type=args.partition, alpha=alpha_val,
            num_clients=args.num_clients,
            student_size=args.student_size,
            num_rounds=rounds, local_epochs=local_ep,
            comm_mode='adaptive', comm_beta=args.comm_beta,
            comm_tau_max=args.comm_tau_max,
            min_upload=args.min_upload,
            num_repeats=repeats,
            output_dir=out_dir,
        )
        r['comm_mode'] = 'adaptive'
        r['comm_tau'] = 0.0
        r['comm_beta'] = args.comm_beta
        r['min_upload'] = args.min_upload
        r['partition'] = args.partition
        r['alpha'] = alpha_val
        results.append(r)
        print(f'  --> adaptive: Top1={r["top1_mean"]*100:.2f}% '
              f'upload_ratio={r.get("upload_ratio_mean", 1.0)*100:.1f}% '
              f'comm_saving={r.get("comm_saving", 0.0)*100:.1f}%')

        df = pd.DataFrame(results)
        df.to_csv('./output/cifar10_afdc_comm_adaptive.csv', index=False)
        print(f'\n实验G完成! 结果: ./output/cifar10_afdc_comm_adaptive.csv')

    # ── 实验B: 异质性纵向扫描 ──
    elif args.exp == 'alpha_scan':
        alphas = [1.0, 0.1, 0.01, 0.001]
        results = []
        for a in alphas:
            print(f'\n{"="*50}\n  AFDC | alpha={a}\n{"="*50}')
            out_dir = f'./output/cifar10_afdc_alpha{a}'
            r = run_with_repeats(
                run_afdc, teacher, train_ds, test_loader, device,
                partition_type='dirichlet', alpha=a,
                num_clients=10, student_size='small',
                num_rounds=rounds, local_epochs=local_ep,
                num_repeats=repeats,
                output_dir=out_dir,
            )
            r['alpha'] = a
            results.append(r)
            print(f'  --> alpha={a}: Top1={r["top1_mean"]*100:.2f}% '
                  f'± {r["top1_std"]*100:.2f}%')

        df = pd.DataFrame(results)
        df.to_csv('./output/cifar10_afdc_alpha_scan.csv', index=False)
        print(f'\n实验B完成!')

    # ── 实验C: 压缩比扫描 ──
    elif args.exp == 'compression':
        COMPRESSION_MAP = {
            'large':  '~3x',
            'medium': '~6x',
            'small':  '~12x',
            'tiny':   '~30x',
        }
        results = []
        for sz in ['large', 'medium', 'small', 'tiny']:
            print(f'\n{"="*50}\n  AFDC | compression={sz} ({COMPRESSION_MAP[sz]})\n{"="*50}')
            out_dir = f'./output/cifar10_afdc_compression_{sz}'
            r = run_with_repeats(
                run_afdc, teacher, train_ds, test_loader, device,
                partition_type='dirichlet', alpha=0.5,
                num_clients=10, student_size=sz,
                num_rounds=rounds, local_epochs=local_ep,
                num_repeats=repeats,
                output_dir=out_dir,
            )
            r['student_size'] = sz
            r['compression_ratio'] = COMPRESSION_MAP[sz]
            results.append(r)
            print(f'  --> {sz} ({COMPRESSION_MAP[sz]}): '
                  f'Top1={r["top1_mean"]*100:.2f}% ± {r["top1_std"]*100:.2f}%')

        df = pd.DataFrame(results)
        df.to_csv('./output/cifar10_afdc_compression.csv', index=False)
        print(f'\n实验C完成!')

    # ── 实验E: 客户端规模扫描 ──
    elif args.exp == 'scale_scan':
        results = []
        for nc in [2, 4, 8, 16]:
            print(f'\n{"="*50}\n  AFDC | clients={nc}\n{"="*50}')
            out_dir = f'./output/cifar10_afdc_clients{nc}'
            r = run_with_repeats(
                run_afdc, teacher, train_ds, test_loader, device,
                partition_type='dirichlet', alpha=0.5,
                num_clients=nc, student_size='small',
                num_rounds=rounds, local_epochs=local_ep,
                num_repeats=repeats,
                output_dir=out_dir,
            )
            r['num_clients'] = nc
            results.append(r)
            print(f'  --> clients={nc}: Top1={r["top1_mean"]*100:.2f}% '
                  f'± {r["top1_std"]*100:.2f}%')

        df = pd.DataFrame(results)
        df.to_csv('./output/cifar10_afdc_scale_scan.csv', index=False)
        print(f'\n实验E完成!')

    # ── 实验F: 消融实验 ──
    elif args.exp == 'ablation':
        ablation_cfgs = [
            ('FedCVAE-Ens', '基准: One-Shot解码器集成',
             {'w_latent': 0.0, 'w_recon': 0.0, 'w_cvae': 0.0,
              'use_adaptive': False, 'num_rounds': 1}),
            ('+Multi-Round-KD', '加入多轮蒸馏(软+硬标签)',
             {'w_latent': 0.0, 'w_recon': 0.0, 'w_cvae': 0.0,
              'use_adaptive': False}),
            ('+Feature-Align', '加入特征对齐损失',
             {'w_latent': 0.3, 'w_recon': 0.2, 'w_cvae': 0.0,
              'use_adaptive': False}),
            ('AFDC-Static', '加入CVAE虚拟数据,静态权重',
             {'w_latent': 0.3, 'w_recon': 0.2, 'w_cvae': 0.1,
              'use_adaptive': False}),
            ('AFDC-Full', '完整AFDC: 动态KL散度自适应调度',
             {'w_latent': 0.3, 'w_recon': 0.2, 'w_cvae': 0.1,
              'use_adaptive': True}),
        ]

        results = []
        for name, desc, cfg in ablation_cfgs:
            for pt in ['iid', 'dirichlet', 'label_skew']:
                print(f'\n{"="*50}\n  Ablation | {name} | {pt}\n  {desc}\n{"="*50}')
                effective_rounds = cfg.get('num_rounds', rounds)

                out_dir = f'./output/cifar10_ablation_{name.replace("+","plus").replace("-","_")}_{pt}'
                r = run_with_repeats(
                    run_afdc, teacher, train_ds, test_loader, device,
                    partition_type=pt, alpha=0.5,
                    num_clients=10, student_size='small',
                    num_rounds=effective_rounds, local_epochs=local_ep,
                    w_latent=cfg['w_latent'],
                    w_recon=cfg['w_recon'],
                    w_cvae=cfg['w_cvae'],
                    use_adaptive=cfg['use_adaptive'],
                    num_repeats=repeats,
                    output_dir=out_dir,
                )
                r['ablation_variant'] = name
                r['partition'] = pt
                results.append(r)
                print(f'  --> {name}/{pt}: Top1={r["top1_mean"]*100:.2f}% '
                      f'± {r["top1_std"]*100:.2f}%')

        df = pd.DataFrame(results)
        df.to_csv('./output/cifar10_ablation.csv', index=False)
        print(f'\n实验F完成! 结果: ./output/cifar10_ablation.csv')

        sep = "=" * 60
        print(f'\n{sep}')
        print(f'  消融实验汇总 (Dirichlet α=0.5)')
        print(f'{sep}')
        for _, row in df[df['partition'] == 'dirichlet'].iterrows():
            variant = row["ablation_variant"]
            m = row["top1_mean"]
            s = row["top1_std"]
            print(f'  {variant:30s}: '
                  f'Top1={m*100:.2f}% ± {s*100:.2f}%')


if __name__ == '__main__':
    main()
