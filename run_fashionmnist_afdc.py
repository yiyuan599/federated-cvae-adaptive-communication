# run_fashionmnist_afdc.py
# -*- coding: utf-8 -*-
"""
AFDC 在 FashionMNIST 上的完整实验脚本（六组对照实验）

使用方法:
  python run_fashionmnist_afdc.py --exp baseline        # 实验A
  python run_fashionmnist_afdc.py --exp alpha_scan      # 实验B
  python run_fashionmnist_afdc.py --exp compression     # 实验C
  python run_fashionmnist_afdc.py --exp superclass      # 实验D
  python run_fashionmnist_afdc.py --exp scale_scan      # 实验E
  python run_fashionmnist_afdc.py --exp ablation        # 实验F
  python run_fashionmnist_afdc.py --exp comm_adaptive   # 实验G
  # 实验H：自适应通信阈值（阈值随全局漂移 EMA 动态缩放）
  python run_fashionmnist_afdc.py --exp comm_adaptive --comm_beta 0.8
  python run_fashionmnist_afdc.py --exp baseline --quick
"""

import os, sys, json, time, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fashionmnist_teacher import (FashionMNISTTeacherCVAE,
    FashionMNISTStudentCVAE, count_parameters, get_model_size_mb)
from datasets.data_partition import FederatedDataPartitioner

LATENT_DIM   = 128
NUM_CLASSES  = 10
NUM_REPEATS  = 3
RANDOM_SEEDS = [42, 123, 999]

# FashionMNIST 归一化常数（单通道）
_FMNIST_MEAN = (0.2860,)
_FMNIST_STD  = (0.3530,)

# FashionMNIST 超类（实验D）
FASHIONMNIST_SUPERCLASSES = {
    0: [0, 6],   # 轻薄上衣: T-shirt, Shirt
    1: [1, 3],   # 下装+裙: Trouser, Dress
    2: [2, 4],   # 厚重上衣: Pullover, Coat
    3: [5, 7],   # 开放式鞋: Sandal, Sneaker
    4: [8, 9],   # 包+靴: Bag, Ankle boot
}


# ============================================================
# 工具函数
# ============================================================
def find_teacher(base='./output/fashionmnist_teacher'):
    path = os.path.join(base, 'teacher_best.pt')
    return path if os.path.exists(path) else None


def load_fashionmnist(data_dir='./data'):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_FMNIST_MEAN, _FMNIST_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_FMNIST_MEAN, _FMNIST_STD),
    ])
    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)
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


def _partition_superclass(train_ds, num_clients, seed=42):
    np.random.seed(seed)
    labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
    label_to_sc = {}
    for sc, cls_list in FASHIONMNIST_SUPERCLASSES.items():
        for c in cls_list:
            label_to_sc[c] = sc
    num_sc = len(FASHIONMNIST_SUPERCLASSES)
    sc_indices = {sc: [] for sc in range(num_sc)}
    for idx, lbl in enumerate(labels):
        sc = label_to_sc.get(int(lbl), -1)
        if sc >= 0:
            sc_indices[sc].append(idx)
    sc_list = list(range(num_sc))
    np.random.shuffle(sc_list)
    scs_per_client = max(1, num_sc // num_clients)
    client_indices = {i: [] for i in range(num_clients)}
    for cid in range(num_clients):
        for j in range(scs_per_client):
            sc = sc_list[(cid * scs_per_client + j) % num_sc]
            idxs = sc_indices[sc][:]
            np.random.shuffle(idxs)
            client_indices[cid].extend(idxs)
        np.random.shuffle(client_indices[cid])
    return client_indices


def partition_data(train_ds, partition_type, num_clients, alpha=0.5, seed=42):
    partitioner = FederatedDataPartitioner(train_ds, num_clients, seed=seed)
    if partition_type == 'iid':
        partition = partitioner.partition_iid()
    elif partition_type == 'dirichlet':
        partition = partitioner.partition_dirichlet(alpha=alpha)
    elif partition_type == 'label_skew':
        partition = partitioner.partition_label_skew(labels_per_client=2)
    elif partition_type == 'superclass':
        partition = _partition_superclass(train_ds, num_clients, seed)
        return [torch.utils.data.Subset(train_ds, partition[i])
                for i in range(num_clients)]
    else:
        raise ValueError(f'未知分区类型: {partition_type}')
    return partitioner.get_client_datasets(partition)


# ============================================================
# CVAE 本地训练
# ============================================================
def train_local_cvae(model, dataset, device, epochs=5, lr=5e-4, beta=0.01,
                     cls_weight=5.0):
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    _mean = torch.tensor([0.2860], device=device).view(1, 1, 1, 1)
    _std  = torch.tensor([0.3530], device=device).view(1, 1, 1, 1)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            recon, mu, logvar, _ = model(x, y)
            x_01 = (x * _std + _mean).clamp(0, 1)
            recon_l = F.mse_loss(recon, x_01)
            kl_l    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            cls_l   = F.cross_entropy(model.classify(mu), y)
            loss = recon_l + beta * kl_l + cls_weight * cls_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


# ============================================================
# 虚拟数据生成
# ============================================================
@torch.no_grad()
def generate_virtual_data(decoder_model, mu_global, sigma_global,
                          device, num_per_class=100):
    decoder_model.eval()
    all_images, all_labels = [], []
    for cls in range(NUM_CLASSES):
        if mu_global is not None and cls in mu_global:
            mu_c    = mu_global[cls].to(device)
            sigma_c = sigma_global[cls].to(device)
            eps     = torch.randn(num_per_class, LATENT_DIM, device=device)
            z       = mu_c.unsqueeze(0) + sigma_c.unsqueeze(0) * eps
        else:
            z = torch.randn(num_per_class, LATENT_DIM, device=device)
        lbl  = torch.full((num_per_class,), cls, dtype=torch.long, device=device)
        imgs = decoder_model.decode(z, lbl).clamp(0, 1)
        # 保持 [0,1]；蒸馏循环中用 x_norm=(x-mean)/std 归一化给编码器
        all_images.append(imgs.cpu())
        all_labels.append(lbl.cpu())
    return torch.cat(all_images), torch.cat(all_labels)


# ============================================================
# 漂移计算
# ============================================================
def compute_drift(mu_stats, sigma_stats, prev_mu, prev_sigma):
    drifts = []
    for c in range(NUM_CLASSES):
        if c in mu_stats and c in prev_mu:
            mu_c, mu_p = mu_stats[c], prev_mu[c]
            sg_c = sigma_stats.get(c, torch.ones(LATENT_DIM) * 0.1)
            sg_p = prev_sigma.get(c, torch.ones(LATENT_DIM) * 0.1)
            var_c, var_p = sg_c**2, sg_p**2
            kl = 0.5 * (((mu_c - mu_p)**2) / var_p.clamp(1e-6) +
                        var_c / var_p.clamp(1e-6) - 1 +
                        torch.log(var_p.clamp(1e-6) / var_c.clamp(1e-6))).mean().item()
            drifts.append(max(0, kl))
    return np.mean(drifts) if drifts else 0.0


# ============================================================
# AFDC 核心
# ============================================================
def run_afdc(teacher, train_ds, test_loader, device,
             partition_type='dirichlet', alpha=0.5, num_clients=10,
             student_size='small', num_rounds=20, local_epochs=5,
             T=4.0, w_latent=0.3, w_recon=0.2, w_cvae=0.1,
             use_adaptive=True, T_check=5,
             comm_mode='full', comm_tau=0.05, min_upload=3,
             comm_beta=0.8, comm_tau_min=0.01, comm_tau_max=1.0,
             num_per_class=100,
             seed=42, output_dir='./output/tmp'):
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 统计提取用确定性测试变换，避免随机增强被误判为潜在分布漂移
    stats_transform = getattr(test_loader.dataset, 'transform', None)

    client_datasets = partition_data(train_ds, partition_type, num_clients, alpha, seed)

    student = FashionMNISTStudentCVAE(
        latent_dim=LATENT_DIM, student_size=student_size).to(device)
    global_model = copy.deepcopy(teacher).to(device)
    global_model.eval()
    client_models = [copy.deepcopy(teacher).to(device) for _ in range(num_clients)]

    _mean = torch.tensor([0.2860], device=device).view(1, 1, 1, 1)
    _std  = torch.tensor([0.3530], device=device).view(1, 1, 1, 1)

    opt_s = optim.Adam(student.parameters(), lr=5e-4)
    scheduler_s = optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=num_rounds)

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

    best_top1 = best_top5 = 0
    history = []

    for round_idx in range(num_rounds):
        print(f'\n--- AFDC Round {round_idx+1}/{num_rounds} ---')

        all_mu_stats, all_sigma_stats, all_class_counts = [], [], []
        round_drift_scores = []

        for cid in range(num_clients):
            class_zs = {c: [] for c in range(NUM_CLASSES)}
            if len(client_datasets[cid]) > 0:
                train_local_cvae(client_models[cid], client_datasets[cid],
                                 device, epochs=local_epochs)
                client_models[cid].eval()
                c_loader = DataLoader(client_datasets[cid], batch_size=256, shuffle=False)
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
            mu_s, sigma_s, count_s = {}, {}, {}
            for c in range(NUM_CLASSES):
                if class_zs[c]:
                    zs = torch.cat(class_zs[c], dim=0)
                    mu_s[c] = zs.mean(dim=0)
                    sigma_s[c] = zs.std(dim=0, unbiased=False).clamp(min=0.01)
                    count_s[c] = zs.shape[0]
                else:
                    mu_s[c] = torch.zeros(LATENT_DIM)
                    sigma_s[c] = torch.ones(LATENT_DIM)
                    count_s[c] = 0
            all_mu_stats.append(mu_s)
            all_sigma_stats.append(sigma_s)
            all_class_counts.append(count_s)

            if (comm_mode in ('sparse', 'adaptive') and round_idx > 0
                    and uploaded_mu_stats[cid] is not None):
                round_drift_scores.append((
                    cid,
                    compute_drift(mu_s, sigma_s, uploaded_mu_stats[cid],
                                  uploaded_sigma_stats[cid])
                ))

            if use_adaptive and round_idx > 0 and round_idx % T_check == 0:
                drift = compute_drift(mu_s, sigma_s, prev_mu_stats[cid],
                                      prev_sigma_stats.get(cid, {}))
                client_drift_history[cid].append(drift)

        # ── 通信调度决策：漂移触发式稀疏上传 ──
        if comm_mode == 'sparse' and round_idx > 0:
            round_drift_scores.sort(key=lambda x: x[1], reverse=True)
            upload_set = {cid for cid, d in round_drift_scores if d >= comm_tau}
            effective_tau = comm_tau
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

        if use_adaptive:
            prev_mu_stats    = {cid: all_mu_stats[cid]    for cid in range(num_clients)}
            prev_sigma_stats = {cid: all_sigma_stats[cid] for cid in range(num_clients)}

        # 聚合全局分布：按每个客户端在该类别上的样本数加权（而非客户端总样本数），
        # 否则某类样本极少的客户端会因其它类别样本多而在该类别上获得不成比例的权重
        mu_global, sigma_global = {}, {}
        for c in range(NUM_CLASSES):
            total_c = sum(uploaded_count_stats[cid][c] for cid in range(num_clients))
            if total_c > 0:
                mu_c = sum(uploaded_mu_stats[cid][c] * uploaded_count_stats[cid][c] / total_c
                           for cid in range(num_clients))
                sg_c = sum(uploaded_sigma_stats[cid][c] * uploaded_count_stats[cid][c] / total_c
                           for cid in range(num_clients))
            else:
                mu_c = torch.zeros(LATENT_DIM)
                sg_c = torch.ones(LATENT_DIM)
            mu_global[c] = mu_c
            sigma_global[c] = sg_c.clamp(min=0.01)

        # 生成虚拟数据
        v_imgs, v_lbls = generate_virtual_data(
            global_model, mu_global, sigma_global, device, num_per_class)
        virtual_ds = torch.utils.data.TensorDataset(v_imgs, v_lbls)
        virtual_loader = DataLoader(virtual_ds, batch_size=64, shuffle=True, drop_last=True)
        print(f'  虚拟数据: {len(virtual_ds)} 样本')

        # 蒸馏
        student.train()
        teacher.eval()
        # 线性渐进权重调度（避免阶跃导致的性能崩溃）
        progress = round_idx / max(1, num_rounds - 1)
        alpha_kd = 0.3 + 0.5 * progress   # 0.3 → 0.8
        alpha_ce = 1.0 - alpha_kd

        if use_adaptive and round_idx > 0 and client_drift_history:
            all_drifts = []
            for cid in range(num_clients):
                all_drifts.extend(client_drift_history[cid][-1:] if client_drift_history[cid] else [0])
            if np.mean(all_drifts) > 0.5:
                alpha_kd = min(alpha_kd + 0.1, 0.9)
            alpha_ce = 1.0 - alpha_kd

        for _ in range(2):
            for x, y in virtual_loader:
                x, y = x.to(device), y.to(device)
                # 虚拟数据来自解码器 [0,1]，编码器训练在归一化 [-∞,+∞] 域
                x_norm = (x - _mean) / _std
                opt_s.zero_grad()

                s_recon, s_mu, s_logvar, _ = student(x_norm, y)
                s_logits = student.classify(s_mu)

                with torch.no_grad():
                    t_recon, t_mu, t_logvar, _ = teacher(x_norm, y)
                    t_logits = teacher.classify(t_mu)

                soft_loss = F.kl_div(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction='batchmean') * (T**2)
                hard_loss  = F.cross_entropy(s_logits, y)
                align_loss = 1.0 - (F.normalize(s_mu, dim=1) *
                                    F.normalize(t_mu, dim=1)).sum(dim=1).mean()
                # s_recon ∈ [0,1]，x（虚拟数据）∈ [0,1]，直接 MSE
                cvae_loss = (F.mse_loss(s_recon, x) +
                             0.01 * (-0.5 * torch.mean(1 + s_logvar
                                    - s_mu.pow(2) - s_logvar.exp())))

                total_loss = (alpha_kd * soft_loss + alpha_ce * hard_loss +
                              w_latent * align_loss + w_cvae * cvae_loss)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_s.step()

        scheduler_s.step()

        top1, top5 = evaluate_topk(student, test_loader, device)
        if top1 > best_top1:
            best_top1, best_top5 = top1, top5
        history.append({'round': round_idx+1, 'top1': top1, 'top5': top5,
                        'best_top1': best_top1,
                        'uploaded_clients': len(upload_set),
                        'upload_ratio': len(upload_set) / num_clients,
                        'comm_bytes': round_comm_bytes,
                        'cum_comm_bytes': cum_comm_bytes,
                        'adaptive_tau': effective_tau,
                        'drift_median': float(np.median(round_drifts)) if round_drifts else 0.0,
                        'drift_min': float(np.min(round_drifts)) if round_drifts else 0.0,
                        'drift_max': float(np.max(round_drifts)) if round_drifts else 0.0})
        print(f'  Round {round_idx+1}: Top1={top1*100:.2f}%, Top5={top5*100:.2f}% '
              f'(best={best_top1*100:.2f}%)')

    pd.DataFrame(history).to_csv(f'{output_dir}/afdc_history.csv', index=False)
    if drift_records:
        pd.DataFrame(drift_records).to_csv(
            f'{output_dir}/drift_history.csv', index=False)
    torch.save(student.state_dict(), f'{output_dir}/afdc_student_final.pt')

    return {
        'method': 'AFDC', 'top1': history[-1]['top1'],
        'top5': history[-1]['top5'], 'best_top1': best_top1,
        'best_top5': best_top5, 'partition': partition_type,
        'student_size': student_size,
        'upload_ratio_mean': np.mean([h['upload_ratio'] for h in history]),
        'comm_bytes_total': cum_comm_bytes,
        'comm_bytes_baseline': num_rounds * num_clients * bytes_per_client,
        'comm_saving': 1.0 - cum_comm_bytes / (
            num_rounds * num_clients * bytes_per_client),
        'history': history,
    }


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
            import traceback; traceback.print_exc()
            results_list.append(None)

    valid = [r for r in results_list if r is not None]
    if not valid:
        raise RuntimeError('所有重复运行均失败!')

    top1s = [r['top1'] for r in valid]
    top5s = [r['top5'] for r in valid]
    best_top1s = [r.get('best_top1', r['top1']) for r in valid]

    summary = {k: valid[0][k] for k in ['method', 'partition', 'student_size']
               if k in valid[0]}
    summary.update({
        'top1_mean': np.mean(top1s), 'top1_std': np.std(top1s),
        'top5_mean': np.mean(top5s), 'top5_std': np.std(top5s),
        'best_top1_mean': np.mean(best_top1s), 'best_top1_std': np.std(best_top1s),
        'top1_individual': top1s, 'top5_individual': top5s,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='baseline',
                        choices=['baseline', 'alpha_scan', 'compression',
                                 'superclass', 'scale_scan', 'ablation',
                                 'comm_adaptive'])
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--min_upload', type=int, default=3,
                        help='稀疏通信保底上传客户端数')
    parser.add_argument('--comm_taus', default=None,
                        help='稀疏通信阈值列表，逗号分隔，如 0.3,0.5')
    parser.add_argument('--comm_beta', type=float, default=0.8,
                        help='自适应阈值缩放系数 (adaptive 模式)')
    parser.add_argument('--comm_tau_max', type=float, default=1.0,
                        help='自适应阈值上限')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    teacher_path = find_teacher()
    if not teacher_path:
        raise FileNotFoundError('请先运行 train_fashionmnist_teacher.py 训练教师模型')

    teacher = FashionMNISTTeacherCVAE(latent_dim=LATENT_DIM).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    print(f'教师模型已加载: {teacher_path}')

    train_ds, test_ds = load_fashionmnist()
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    repeats     = 1   if args.quick else NUM_REPEATS
    num_rounds  = 5   if args.quick else 20
    local_ep    = 1   if args.quick else 5
    npc         = 20  if args.quick else 100  # num_per_class

    results = []

    if args.exp == 'baseline':
        for pt in ['iid', 'dirichlet', 'label_skew']:
            print(f'\n{"="*50}\n  AFDC | {pt}\n{"="*50}')
            out_dir = f'./output/fashionmnist_afdc_{pt}'
            r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                                 partition_type=pt, alpha=args.alpha,
                                 num_clients=args.num_clients,
                                 num_rounds=num_rounds, local_epochs=local_ep,
                                 num_per_class=npc,
                                 num_repeats=repeats, output_dir=out_dir)
            r['partition'] = pt
            print(f'  --> {pt}: Top1={r["top1_mean"]*100:.2f}% ± {r["top1_std"]*100:.2f}%')
            results.append(r)

        df = pd.DataFrame(results)
        df.to_csv('./output/fashionmnist_afdc_baseline.csv', index=False)

    elif args.exp == 'comm_adaptive':
        tau_values = ([float(x) for x in args.comm_taus.split(',')]
                      if args.comm_taus else
                      ([0.05, 0.1, 0.2] if not args.quick else [0.1]))

        print(f'\n{"="*50}\n  Comm | full | dirichlet\n{"="*50}')
        out_dir = './output/fashionmnist_comm_full_dirichlet'
        r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                             partition_type='dirichlet', alpha=0.5,
                             num_clients=args.num_clients,
                             num_rounds=num_rounds, local_epochs=local_ep,
                             num_per_class=npc, comm_mode='full',
                             num_repeats=repeats, output_dir=out_dir)
        r.update({'comm_mode': 'full', 'comm_tau': 0.0,
                  'min_upload': args.min_upload, 'partition': 'dirichlet'})
        results.append(r)
        print(f'  --> full: Top1={r["top1_mean"]*100:.2f}% '
              f'comm_saving={r.get("comm_saving", 0.0)*100:.1f}%')

        for tau in tau_values:
            print(f'\n{"="*50}\n  Comm | sparse tau={tau} | dirichlet\n{"="*50}')
            out_dir = (f'./output/fashionmnist_comm_sparse_tau{tau}_'
                       f'mu{args.min_upload}_dirichlet')
            r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                                 partition_type='dirichlet', alpha=0.5,
                                 num_clients=args.num_clients,
                                 num_rounds=num_rounds, local_epochs=local_ep,
                                 num_per_class=npc,
                                 comm_mode='sparse', comm_tau=tau,
                                 min_upload=args.min_upload,
                                 num_repeats=repeats, output_dir=out_dir)
            r.update({'comm_mode': 'sparse', 'comm_tau': tau,
                      'min_upload': args.min_upload, 'partition': 'dirichlet'})
            results.append(r)
            print(f'  --> sparse tau={tau}: Top1={r["top1_mean"]*100:.2f}% '
                  f'upload_ratio={r.get("upload_ratio_mean", 1.0)*100:.1f}% '
                  f'comm_saving={r.get("comm_saving", 0.0)*100:.1f}%')

        print(f'\n{"="*50}\n  Comm | adaptive | dirichlet\n{"="*50}')
        out_dir = './output/fashionmnist_comm_adaptive_dirichlet'
        r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                             partition_type='dirichlet', alpha=0.5,
                             num_clients=args.num_clients,
                             num_rounds=num_rounds, local_epochs=local_ep,
                             num_per_class=npc,
                             comm_mode='adaptive', comm_beta=args.comm_beta,
                             comm_tau_max=args.comm_tau_max,
                             min_upload=args.min_upload,
                             num_repeats=repeats, output_dir=out_dir)
        r.update({'comm_mode': 'adaptive', 'comm_tau': 0.0,
                  'comm_beta': args.comm_beta,
                  'min_upload': args.min_upload, 'partition': 'dirichlet'})
        results.append(r)
        print(f'  --> adaptive: Top1={r["top1_mean"]*100:.2f}% '
              f'upload_ratio={r.get("upload_ratio_mean", 1.0)*100:.1f}% '
              f'comm_saving={r.get("comm_saving", 0.0)*100:.1f}%')

        pd.DataFrame(results).to_csv(
            './output/fashionmnist_afdc_comm_adaptive.csv', index=False)
        print(f'\n实验G完成! 结果: ./output/fashionmnist_afdc_comm_adaptive.csv')

    elif args.exp == 'alpha_scan':
        alphas = [1.0, 0.1, 0.01, 0.001] if not args.quick else [1.0, 0.1]
        for a in alphas:
            out_dir = f'./output/fashionmnist_afdc_alpha{a}'
            r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                                 partition_type='dirichlet', alpha=a,
                                 num_clients=10, num_rounds=num_rounds,
                                 local_epochs=local_ep, num_per_class=npc,
                                 num_repeats=repeats, output_dir=out_dir)
            r['alpha'] = a
            results.append(r)
        pd.DataFrame(results).to_csv('./output/fashionmnist_afdc_alpha_scan.csv', index=False)

    elif args.exp == 'compression':
        # 不同潜在维度模拟压缩比
        for ldim in ([64, 128, 256] if not args.quick else [64, 128]):
            teacher_c = FashionMNISTTeacherCVAE(latent_dim=ldim).to(device)
            teacher_c.load_state_dict(torch.load(teacher_path, map_location=device),
                                      strict=False)
            teacher_c.eval()
            out_dir = f'./output/fashionmnist_afdc_latent{ldim}'
            r = run_with_repeats(run_afdc, teacher_c, train_ds, test_loader, device,
                                 partition_type='dirichlet', alpha=0.5,
                                 num_clients=10, num_rounds=num_rounds,
                                 local_epochs=local_ep, num_per_class=npc,
                                 num_repeats=repeats, output_dir=out_dir)
            r['latent_dim'] = ldim
            results.append(r)
        pd.DataFrame(results).to_csv('./output/fashionmnist_afdc_compression.csv', index=False)

    elif args.exp == 'superclass':
        print(f'\n{"="*50}\n  AFDC | superclass\n{"="*50}')
        out_dir = './output/fashionmnist_afdc_superclass'
        r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                             partition_type='superclass', alpha=0.5,
                             num_clients=args.num_clients,
                             num_rounds=num_rounds, local_epochs=local_ep,
                             num_per_class=npc,
                             num_repeats=repeats, output_dir=out_dir)
        r['partition'] = 'superclass'
        results.append(r)
        pd.DataFrame(results).to_csv('./output/fashionmnist_afdc_superclass.csv', index=False)

    elif args.exp == 'scale_scan':
        client_counts = [2, 4, 8, 16] if not args.quick else [2, 4]
        for nc in client_counts:
            out_dir = f'./output/fashionmnist_afdc_clients{nc}'
            r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                                 partition_type='dirichlet', alpha=0.5,
                                 num_clients=nc, num_rounds=num_rounds,
                                 local_epochs=local_ep, num_per_class=npc,
                                 num_repeats=repeats, output_dir=out_dir)
            r['num_clients'] = nc
            results.append(r)
        pd.DataFrame(results).to_csv('./output/fashionmnist_afdc_scale_scan.csv', index=False)

    elif args.exp == 'ablation':
        # 消融：逐步去掉 AFDC 组件
        configs = [
            {'use_adaptive': True,  'w_latent': 0.3, 'w_recon': 0.2, 'w_cvae': 0.1,
             'label': 'full_afdc'},
            {'use_adaptive': False, 'w_latent': 0.3, 'w_recon': 0.2, 'w_cvae': 0.1,
             'label': 'no_adaptive'},
            {'use_adaptive': True,  'w_latent': 0.0, 'w_recon': 0.2, 'w_cvae': 0.1,
             'label': 'no_align'},
            {'use_adaptive': True,  'w_latent': 0.3, 'w_recon': 0.0, 'w_cvae': 0.0,
             'label': 'no_cvae'},
            {'use_adaptive': False, 'w_latent': 0.0, 'w_recon': 0.0, 'w_cvae': 0.0,
             'label': 'kd_only'},
        ]
        for cfg in configs:
            label = cfg.pop('label')
            for pt in ['iid', 'dirichlet', 'label_skew']:
                out_dir = f'./output/fashionmnist_ablation_{label}_{pt}'
                r = run_with_repeats(run_afdc, teacher, train_ds, test_loader, device,
                                     partition_type=pt, alpha=0.5,
                                     num_clients=10, num_rounds=num_rounds,
                                     local_epochs=local_ep, num_per_class=npc,
                                     num_repeats=repeats, output_dir=out_dir,
                                     **cfg)
                r['ablation_config'] = label
                r['partition'] = pt
                results.append(r)
        pd.DataFrame(results).to_csv('./output/fashionmnist_ablation.csv', index=False)


if __name__ == '__main__':
    main()
