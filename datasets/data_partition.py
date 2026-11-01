# -*- coding: utf-8 -*-
"""
联邦学习数据分布工具
实现3种异构数据分布：
1. IID (独立同分布)
2. Dirichlet Non-IID (基于Dirichlet分布的异构)
3. Label-skew Non-IID (标签倾斜异构)

用于模拟真实联邦学习场景中的数据异构性
"""

import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


class FederatedDataPartitioner:
    """
    联邦学习数据分区器
    将数据集按不同策略分配给多个客户端
    """
    
    def __init__(self, dataset: Dataset, num_clients: int, seed: int = 42):
        """
        初始化数据分区器
        
        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            seed: 随机种子
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        
        # 获取标签
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.labels = np.array(dataset.labels)
        else:
            # 遍历获取标签
            self.labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        self.num_classes = len(np.unique(self.labels))
        self.num_samples = len(self.labels)
        
        print(f"数据集信息:")
        print(f"  样本数: {self.num_samples}")
        print(f"  类别数: {self.num_classes}")
        print(f"  客户端数: {self.num_clients}")
    
    def partition_iid(self) -> Dict[int, List[int]]:
        """
        IID分区：每个客户端获得相同数量且分布均匀的数据
        
        Returns:
            Dict[int, List[int]]: 客户端ID -> 样本索引列表
        """
        np.random.seed(self.seed)
        
        # 打乱所有索引
        indices = np.random.permutation(self.num_samples)
        
        # 平均分配
        split_size = self.num_samples // self.num_clients
        client_indices = {}
        
        for i in range(self.num_clients):
            start = i * split_size
            end = start + split_size if i < self.num_clients - 1 else self.num_samples
            client_indices[i] = indices[start:end].tolist()
        
        self._print_distribution(client_indices, "IID")
        return client_indices
    
    def partition_dirichlet(self, alpha: float = 0.5) -> Dict[int, List[int]]:
        """
        Dirichlet Non-IID分区：使用Dirichlet分布控制异构程度
        
        Args:
            alpha: Dirichlet参数，越小异构程度越高
                   - alpha=0.1: 高度异构
                   - alpha=0.5: 中度异构
                   - alpha=1.0: 轻度异构
                   - alpha->∞: 接近IID
        
        Returns:
            Dict[int, List[int]]: 客户端ID -> 样本索引列表
        """
        np.random.seed(self.seed)
        
        client_indices = {i: [] for i in range(self.num_clients)}
        
        # 按类别处理
        for class_id in range(self.num_classes):
            class_indices = np.where(self.labels == class_id)[0]
            np.random.shuffle(class_indices)
            
            # Dirichlet分布决定每个客户端分到的比例
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            
            # 按比例分配
            proportions = (proportions * len(class_indices)).astype(int)
            
            # 确保总数正确
            diff = len(class_indices) - proportions.sum()
            for i in range(abs(diff)):
                proportions[i % self.num_clients] += np.sign(diff)
            
            # 分配样本
            start = 0
            for client_id, count in enumerate(proportions):
                client_indices[client_id].extend(class_indices[start:start + count].tolist())
                start += count
        
        # 打乱每个客户端内部的顺序
        for client_id in client_indices:
            np.random.shuffle(client_indices[client_id])
        
        self._print_distribution(client_indices, f"Dirichlet (alpha={alpha})")
        return client_indices
    
    def partition_label_skew(self, labels_per_client: int = 2) -> Dict[int, List[int]]:
        """
        Label-skew Non-IID分区：每个客户端只拥有部分类别的数据
        
        Args:
            labels_per_client: 每个客户端拥有的类别数
        
        Returns:
            Dict[int, List[int]]: 客户端ID -> 样本索引列表
        """
        np.random.seed(self.seed)
        
        if labels_per_client > self.num_classes:
            labels_per_client = self.num_classes
        
        client_indices = {i: [] for i in range(self.num_clients)}
        
        # 按类别整理索引
        class_indices = {c: np.where(self.labels == c)[0].tolist() for c in range(self.num_classes)}
        
        # 为每个客户端分配类别
        # 使用循环策略确保每个类别被多个客户端覆盖
        for client_id in range(self.num_clients):
            # 选择该客户端的类别（循环分配）
            start_class = (client_id * labels_per_client) % self.num_classes
            client_classes = [(start_class + i) % self.num_classes for i in range(labels_per_client)]
            
            for class_id in client_classes:
                if class_indices[class_id]:
                    # 从该类别中随机取一部分样本
                    num_samples = len(class_indices[class_id]) // (self.num_clients // (self.num_classes // labels_per_client) + 1)
                    num_samples = max(1, num_samples)
                    
                    selected = class_indices[class_id][:num_samples]
                    client_indices[client_id].extend(selected)
                    class_indices[class_id] = class_indices[class_id][num_samples:]
        
        # 处理剩余样本（分配给样本最少的客户端）
        remaining = []
        for class_id, indices in class_indices.items():
            remaining.extend(indices)
        
        np.random.shuffle(remaining)
        client_sizes = [len(client_indices[i]) for i in range(self.num_clients)]
        
        for idx in remaining:
            min_client = np.argmin(client_sizes)
            client_indices[min_client].append(idx)
            client_sizes[min_client] += 1
        
        # 打乱每个客户端内部的顺序
        for client_id in client_indices:
            np.random.shuffle(client_indices[client_id])
        
        self._print_distribution(client_indices, f"Label-skew ({labels_per_client} labels/client)")
        return client_indices
    
    def get_client_datasets(self, partition: Dict[int, List[int]]) -> List[Subset]:
        """
        根据分区结果创建客户端数据集

        Args:
            partition: 分区结果

        Returns:
            List[Subset]: 客户端数据集列表
        """
        return [Subset(self.dataset, partition[i]) for i in range(self.num_clients)]

    # ── CIFAR-100 超类结构定义 ─────────────────────────────────────
    CIFAR100_SUPERCLASSES = {
        0:  [4, 30, 55, 72, 95],   # 水生哺乳动物
        1:  [1, 32, 67, 73, 91],   # 鱼类
        2:  [54, 62, 70, 82, 92],  # 花卉
        3:  [9, 10, 16, 28, 61],   # 食品容器
        4:  [0, 51, 53, 57, 83],   # 水果蔬菜
        5:  [22, 39, 40, 86, 87],  # 家用电器
        6:  [5, 20, 25, 84, 94],   # 家具
        7:  [6, 7, 14, 18, 24],    # 昆虫
        8:  [3, 42, 43, 88, 97],   # 大型食肉动物
        9:  [12, 17, 37, 68, 76],  # 大型人造户外物品
        10: [23, 33, 49, 60, 71],  # 大型自然户外场景
        11: [15, 19, 21, 31, 38],  # 大型杂食动物/食草动物
        12: [34, 63, 64, 66, 75],  # 中型哺乳动物
        13: [26, 45, 77, 79, 99],  # 非昆虫无脊椎动物
        14: [2, 11, 35, 46, 98],   # 人
        15: [27, 29, 44, 78, 93],  # 爬行动物
        16: [36, 50, 65, 74, 80],  # 小型哺乳动物
        17: [47, 52, 56, 59, 96],  # 树木
        18: [8, 13, 48, 58, 90],   # 交通工具1
        19: [41, 69, 81, 85, 89],  # 交通工具2
    }

    @classmethod
    def _get_subclass_to_superclass(cls):
        mapping = {}
        for sc_idx, subclasses in cls.CIFAR100_SUPERCLASSES.items():
            for sc in subclasses:
                mapping[sc] = sc_idx
        return mapping

    def partition_superclass_skew(self, superclasses_per_client: int = 2) -> Dict[int, List[int]]:
        """
        超类偏斜分区（实验D —— CIFAR-100独有）
        每个客户端只分配 superclasses_per_client 个超类的全部子类数据。
        默认 superclasses_per_client=2，10个客户端恰好覆盖全部20个超类。

        Args:
            superclasses_per_client: 每个客户端持有的超类数量（默认2）

        Returns:
            Dict[int, List[int]]: 客户端ID -> 样本索引列表
        """
        np.random.seed(self.seed)
        subclass_to_sc = self._get_subclass_to_superclass()

        # 按超类聚合样本下标
        superclass_indices = {i: [] for i in range(20)}
        for idx, lbl in enumerate(self.labels):
            sc = subclass_to_sc.get(int(lbl), -1)
            if sc >= 0:
                superclass_indices[sc].append(idx)

        # 打乱超类列表
        superclass_list = list(range(20))
        np.random.shuffle(superclass_list)

        client_indices = {i: [] for i in range(self.num_clients)}
        for client_id in range(self.num_clients):
            for j in range(superclasses_per_client):
                sc_idx = superclass_list[
                    (client_id * superclasses_per_client + j) % 20
                ]
                indices = superclass_indices[sc_idx][:]
                np.random.shuffle(indices)
                client_indices[client_id].extend(indices)

        # 打乱每个客户端内部顺序
        for client_id in client_indices:
            np.random.shuffle(client_indices[client_id])

        self._print_distribution(client_indices,
            f"Superclass-skew ({superclasses_per_client} superclasses/client)")
        return client_indices

    def partition_subclass_skew(self, subclasses_per_client: int = 10) -> Dict[int, List[int]]:
        """
        子类偏斜分区（实验D对照组）
        每个客户端分配 subclasses_per_client 个子类，每个子类来自不同的超类。
        保证各客户端超类覆盖相似但具体子类不同，形成"温和的语义偏斜"。

        Args:
            subclasses_per_client: 每个客户端持有的子类数量（默认10）

        Returns:
            Dict[int, List[int]]: 客户端ID -> 样本索引列表
        """
        np.random.seed(self.seed)
        subclass_to_sc = self._get_subclass_to_superclass()

        # 按子类聚合样本下标
        class_indices = {c: np.where(self.labels == c)[0].tolist()
                        for c in range(self.num_classes)}

        # 按超类组织子类列表
        superclass_subclasses = {sc: subs[:] for sc, subs in self.CIFAR100_SUPERCLASSES.items()}
        for subs in superclass_subclasses.values():
            np.random.shuffle(subs)

        client_indices = {i: [] for i in range(self.num_clients)}

        for client_id in range(self.num_clients):
            assigned = []
            # 从20个超类中轮流取子类（每个超类取一个）
            sc_order = list(range(20))
            np.random.shuffle(sc_order)
            offset = client_id  # 每个客户端从不同的索引开始，避免所有客户端拿到相同子类
            for k in range(20):
                sc = sc_order[(offset + k) % 20]
                if len(assigned) >= subclasses_per_client:
                    break
                subs = superclass_subclasses[sc]
                pick_idx = client_id % len(subs)
                sub = subs[pick_idx]
                if sub not in assigned:
                    assigned.append(sub)

            # 如果还不够 subclasses_per_client 个（20<subclasses_per_client），从剩余子类补齐
            if len(assigned) < subclasses_per_client:
                all_remaining = [c for c in range(self.num_classes)
                               if c not in assigned]
                np.random.shuffle(all_remaining)
                assigned.extend(all_remaining[:subclasses_per_client - len(assigned)])

            # 收集这些子类的样本
            for cls in assigned:
                indices = class_indices.get(cls, [])
                client_indices[client_id].extend(indices)

        # 打乱每个客户端内部顺序
        for cid in client_indices:
            np.random.shuffle(client_indices[cid])

        self._print_distribution(client_indices,
            f"Subclass-skew ({subclasses_per_client} subclasses/client)")
        return client_indices
    
    def _print_distribution(self, partition: Dict[int, List[int]], name: str):
        """打印数据分布统计"""
        print(f"\n{name} 分区统计:")
        print("-" * 50)
        
        for client_id in range(self.num_clients):
            indices = partition[client_id]
            client_labels = self.labels[indices]
            
            # 类别分布
            unique, counts = np.unique(client_labels, return_counts=True)
            dist_str = ", ".join([f"{c}:{cnt}" for c, cnt in zip(unique, counts)])
            
            print(f"  客户端 {client_id}: {len(indices)} 样本, 类别分布: {dist_str}")
        
        print("-" * 50)
    
    def visualize_distribution(self, partition: Dict[int, List[int]], title: str = "Data Distribution", save_path: str = None):
        """
        可视化数据分布
        
        Args:
            partition: 分区结果
            title: 图表标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 构建分布矩阵
        dist_matrix = np.zeros((self.num_clients, self.num_classes))
        
        for client_id in range(self.num_clients):
            indices = partition[client_id]
            client_labels = self.labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            for c, cnt in zip(unique, counts):
                dist_matrix[client_id, c] = cnt
        
        # 绘制堆叠条形图
        bottom = np.zeros(self.num_clients)
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        
        for class_id in range(self.num_classes):
            ax.bar(range(self.num_clients), dist_matrix[:, class_id], 
                   bottom=bottom, label=f'Class {class_id}', color=colors[class_id])
            bottom += dist_matrix[:, class_id]
        
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticks(range(self.num_clients))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"分布图已保存: {save_path}")
        
        plt.close()
        return fig


def create_federated_datasets(dataset: Dataset, 
                               num_clients: int, 
                               partition_type: str = 'iid',
                               **kwargs) -> Tuple[List[Subset], Dict[int, List[int]]]:
    """
    便捷函数：创建联邦学习数据集
    
    Args:
        dataset: 原始数据集
        num_clients: 客户端数量
        partition_type: 分区类型 ('iid', 'dirichlet', 'label_skew')
        **kwargs: 分区参数
            - alpha: Dirichlet参数（用于dirichlet分区）
            - labels_per_client: 每客户端类别数（用于label_skew分区）
    
    Returns:
        Tuple[List[Subset], Dict]: 客户端数据集列表和分区索引
    """
    partitioner = FederatedDataPartitioner(dataset, num_clients)
    
    if partition_type == 'iid':
        partition = partitioner.partition_iid()
    elif partition_type == 'dirichlet':
        alpha = kwargs.get('alpha', 0.5)
        partition = partitioner.partition_dirichlet(alpha=alpha)
    elif partition_type == 'label_skew':
        labels_per_client = kwargs.get('labels_per_client', 2)
        partition = partitioner.partition_label_skew(labels_per_client=labels_per_client)
    elif partition_type == 'superclass':
        superclasses_per_client = kwargs.get('superclasses_per_client', 2)
        partition = partitioner.partition_superclass_skew(
            superclasses_per_client=superclasses_per_client)
    elif partition_type == 'subclass_skew':
        subclasses_per_client = kwargs.get('subclasses_per_client', 10)
        partition = partitioner.partition_subclass_skew(
            subclasses_per_client=subclasses_per_client)
    else:
        raise ValueError(f"未知的分区类型: {partition_type}")
    
    client_datasets = partitioner.get_client_datasets(partition)
    
    return client_datasets, partition


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    from torchvision import datasets, transforms
    
    # 加载CIFAR-10
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    
    # 测试3种分区
    partitioner = FederatedDataPartitioner(dataset, num_clients=5)
    
    print("\n" + "=" * 60)
    print("测试 IID 分区")
    print("=" * 60)
    iid_partition = partitioner.partition_iid()
    partitioner.visualize_distribution(iid_partition, "IID Distribution", "output/iid_dist.png")
    
    print("\n" + "=" * 60)
    print("测试 Dirichlet Non-IID 分区 (alpha=0.5)")
    print("=" * 60)
    dir_partition = partitioner.partition_dirichlet(alpha=0.5)
    partitioner.visualize_distribution(dir_partition, "Dirichlet Non-IID (alpha=0.5)", "output/dirichlet_dist.png")
    
    print("\n" + "=" * 60)
    print("测试 Label-skew Non-IID 分区 (2 labels/client)")
    print("=" * 60)
    skew_partition = partitioner.partition_label_skew(labels_per_client=2)
    partitioner.visualize_distribution(skew_partition, "Label-skew Non-IID (2 labels)", "output/label_skew_dist.png")
    
    print("\n测试完成!")

