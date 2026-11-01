# -*- coding: utf-8 -*-
"""
数据集加载模块
支持CIFAR-10, CIFAR-100
支持3种数据分布: IID, Dirichlet Non-IID, Label-skew Non-IID
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import json


class FederatedDataset:
    """联邦学习数据集管理器"""
    
    SUPPORTED_DATASETS = ['cifar10', 'cifar100', 'mnist', 'fashionmnist']
    SUPPORTED_PARTITIONS = ['iid', 'dirichlet', 'label_skew']
    
    def __init__(self, dataset_name='cifar10', data_dir='./data', 
                 num_clients=10, partition='iid', **kwargs):
        """
        Args:
            dataset_name: 数据集名称 ('cifar10', 'cifar100', 'mnist')
            data_dir: 数据存储目录
            num_clients: 客户端数量
            partition: 数据划分方式 ('iid', 'dirichlet', 'label_skew')
            **kwargs: 其他参数
                - alpha: Dirichlet分布参数 (default: 0.5)
                - labels_per_client: 每个客户端的标签数量 (用于label_skew)
        """
        assert dataset_name in self.SUPPORTED_DATASETS
        assert partition in self.SUPPORTED_PARTITIONS
        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.partition = partition
        self.alpha = kwargs.get('alpha', 0.5)
        self.labels_per_client = kwargs.get('labels_per_client', 2)
        
        # 加载数据集
        self.train_dataset, self.test_dataset = self._load_dataset()
        self.num_classes = self._get_num_classes()
        
        # 数据划分
        self.client_indices = self._partition_data()
        
        # 图像参数
        self.img_size, self.img_channels = self._get_img_params()
    
    def _load_dataset(self):
        """加载数据集"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        if self.dataset_name == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform_train)
            test = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform_test)
            
        elif self.dataset_name == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            train = datasets.CIFAR100(self.data_dir, train=True, download=True, transform=transform_train)
            test = datasets.CIFAR100(self.data_dir, train=False, download=True, transform=transform_test)
            
        elif self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.Resize(32),  # 统一为32x32
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
            test = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)

        elif self.dataset_name == 'fashionmnist':
            transform = transforms.Compose([
                transforms.Resize(32),  # 统一为32x32
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            train = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=transform)
            test = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=transform)

        return train, test
    
    def _get_num_classes(self):
        """获取类别数量"""
        if self.dataset_name == 'cifar10':
            return 10
        elif self.dataset_name == 'cifar100':
            return 100
        elif self.dataset_name in ('mnist', 'fashionmnist'):
            return 10

    def _get_img_params(self):
        """获取图像参数"""
        if self.dataset_name in ['cifar10', 'cifar100']:
            return 32, 3
        elif self.dataset_name in ('mnist', 'fashionmnist'):
            return 32, 1  # 已resize为32x32
    
    def _partition_data(self):
        """数据划分"""
        labels = np.array(self.train_dataset.targets)
        
        if self.partition == 'iid':
            return self._iid_partition(labels)
        elif self.partition == 'dirichlet':
            return self._dirichlet_partition(labels)
        elif self.partition == 'label_skew':
            return self._label_skew_partition(labels)
    
    def _iid_partition(self, labels):
        """IID划分: 随机均匀分配"""
        n_samples = len(labels)
        indices = np.random.permutation(n_samples)
        client_indices = np.array_split(indices, self.num_clients)
        return {i: idx.tolist() for i, idx in enumerate(client_indices)}
    
    def _dirichlet_partition(self, labels):
        """Dirichlet Non-IID划分"""
        n_classes = self.num_classes
        label_indices = defaultdict(list)
        
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        client_indices = {i: [] for i in range(self.num_clients)}
        
        for label in range(n_classes):
            indices = np.array(label_indices[label])
            np.random.shuffle(indices)
            
            # Dirichlet分布
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            split_indices = np.split(indices, proportions)
            
            for client_id, split in enumerate(split_indices):
                client_indices[client_id].extend(split.tolist())
        
        return client_indices
    
    def _label_skew_partition(self, labels):
        """标签偏斜Non-IID划分: 每个客户端只有部分标签"""
        n_classes = self.num_classes
        label_indices = defaultdict(list)
        
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        # 为每个客户端分配标签
        client_labels = {}
        all_labels = list(range(n_classes))
        
        for i in range(self.num_clients):
            # 循环分配标签，确保覆盖所有标签
            start = (i * self.labels_per_client) % n_classes
            client_labels[i] = [(start + j) % n_classes for j in range(self.labels_per_client)]
        
        client_indices = {i: [] for i in range(self.num_clients)}
        
        # 统计每个标签被多少客户端持有
        label_client_count = defaultdict(int)
        for i, labels_list in client_labels.items():
            for label in labels_list:
                label_client_count[label] += 1
        
        # 分配数据
        for label in range(n_classes):
            indices = np.array(label_indices[label])
            np.random.shuffle(indices)
            
            # 获取持有该标签的客户端
            holders = [i for i, labels_list in client_labels.items() if label in labels_list]
            
            # 均匀分配给这些客户端
            splits = np.array_split(indices, len(holders))
            for client_id, split in zip(holders, splits):
                client_indices[client_id].extend(split.tolist())
        
        return client_indices
    
    def get_client_dataloader(self, client_id, batch_size=32, shuffle=True):
        """获取客户端的数据加载器"""
        indices = self.client_indices[client_id]
        subset = Subset(self.train_dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    def get_test_dataloader(self, batch_size=128):
        """获取测试集数据加载器"""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def get_client_stats(self):
        """获取各客户端的数据统计"""
        stats = {}
        labels = np.array(self.train_dataset.targets)
        
        for client_id, indices in self.client_indices.items():
            client_labels = labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            stats[client_id] = {
                'num_samples': len(indices),
                'label_distribution': dict(zip(unique.tolist(), counts.tolist()))
            }
        
        return stats
    
    def save_partition_info(self, save_dir):
        """保存划分信息"""
        os.makedirs(save_dir, exist_ok=True)
        
        info = {
            'dataset': self.dataset_name,
            'num_clients': self.num_clients,
            'partition': self.partition,
            'alpha': self.alpha if self.partition == 'dirichlet' else None,
            'labels_per_client': self.labels_per_client if self.partition == 'label_skew' else None,
            'num_classes': self.num_classes,
            'client_stats': self.get_client_stats()
        }
        
        with open(os.path.join(save_dir, 'partition_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"划分信息已保存到 {save_dir}")


def visualize_distribution(fed_dataset, save_path=None):
    """可视化数据分布"""
    import matplotlib.pyplot as plt
    
    stats = fed_dataset.get_client_stats()
    num_clients = fed_dataset.num_clients
    num_classes = fed_dataset.num_classes
    
    # 创建分布矩阵
    dist_matrix = np.zeros((num_clients, num_classes))
    for client_id, client_stats in stats.items():
        for label, count in client_stats['label_distribution'].items():
            dist_matrix[client_id, label] = count
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.5), max(8, num_clients * 0.4)))
    im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Client ID')
    ax.set_title(f'{fed_dataset.dataset_name.upper()} - {fed_dataset.partition.upper()} Distribution')
    
    # 设置刻度
    ax.set_xticks(range(0, num_classes, max(1, num_classes // 10)))
    ax.set_yticks(range(num_clients))
    
    plt.colorbar(im, ax=ax, label='Number of samples')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"分布图已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("数据集加载测试")
    print("=" * 60)
    
    # 测试不同数据集和划分方式
    configs = [
        {'dataset_name': 'cifar10', 'partition': 'iid'},
        {'dataset_name': 'cifar10', 'partition': 'dirichlet', 'alpha': 0.5},
        {'dataset_name': 'cifar10', 'partition': 'label_skew', 'labels_per_client': 2},
        {'dataset_name': 'cifar100', 'partition': 'dirichlet', 'alpha': 0.1},
    ]
    
    for config in configs:
        print(f"\n{'='*20} {config['dataset_name']} - {config['partition']} {'='*20}")
        
        fed_data = FederatedDataset(
            num_clients=10,
            data_dir='./data',
            **config
        )
        
        print(f"数据集: {fed_data.dataset_name}")
        print(f"类别数: {fed_data.num_classes}")
        print(f"图像大小: {fed_data.img_size}x{fed_data.img_size}x{fed_data.img_channels}")
        print(f"划分方式: {fed_data.partition}")
        
        stats = fed_data.get_client_stats()
        print("\n客户端数据分布:")
        for client_id in range(min(3, fed_data.num_clients)):
            client_stats = stats[client_id]
            print(f"  Client {client_id}: {client_stats['num_samples']} samples, "
                  f"{len(client_stats['label_distribution'])} classes")
        
        # 测试数据加载
        loader = fed_data.get_client_dataloader(0, batch_size=32)
        batch = next(iter(loader))
        print(f"\n批次测试:")
        print(f"  图像: {batch[0].shape}")
        print(f"  标签: {batch[1].shape}")

