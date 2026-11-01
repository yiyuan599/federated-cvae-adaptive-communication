# -*- coding: utf-8 -*-
"""
大规模CVAE模型实现
支持10M/50M/100M+参数规模
适用于CIFAR-10/CIFAR-100等更复杂数据集

模型规模:
- Small: ~10M参数
- Medium: ~50M参数
- Large: ~100M参数 (预留)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = out + residual
        return F.relu(out)


class LargeCVAE(nn.Module):
    """
    大规模条件变分自编码器
    支持不同参数规模配置
    
    模型规模通过 model_size 参数控制:
    - 'small': ~10M参数
    - 'medium': ~50M参数
    - 'large': ~100M参数
    """
    
    # 模型配置
    MODEL_CONFIGS = {
        'small': {
            'encoder_channels': [64, 128, 256, 512],
            'latent_dim': 256,
            'hidden_dim': 1024,
            'num_res_blocks': 2,
        },
        'medium': {
            'encoder_channels': [128, 256, 512, 1024],
            'latent_dim': 512,
            'hidden_dim': 2048,
            'num_res_blocks': 3,
        },
        'large': {
            'encoder_channels': [256, 512, 1024, 2048],
            'latent_dim': 1024,
            'hidden_dim': 4096,
            'num_res_blocks': 4,
        }
    }
    
    def __init__(self, img_size=32, img_channels=3, num_classes=10, 
                 model_size='small', use_bn=True):
        """
        Args:
            img_size: 图像尺寸 (32 for CIFAR)
            img_channels: 图像通道数 (3 for CIFAR)
            num_classes: 类别数量 (10 for CIFAR-10, 100 for CIFAR-100)
            model_size: 模型规模 ('small', 'medium', 'large')
            use_bn: 是否使用BatchNorm
        """
        super().__init__()
        
        assert model_size in self.MODEL_CONFIGS, f"不支持的模型规模: {model_size}"
        config = self.MODEL_CONFIGS[model_size]
        
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.model_size = model_size
        self.use_bn = use_bn
        
        self.encoder_channels = config['encoder_channels']
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_res_blocks = config['num_res_blocks']
        
        # 计算编码后的特征图尺寸 (每次下采样除以2)
        self.encoded_size = img_size // (2 ** len(self.encoder_channels))
        self.encoded_channels = self.encoder_channels[-1]
        
        # ============ 编码器 ============
        self._build_encoder()
        
        # ============ 潜在空间 ============
        # 展平后的维度 + 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, self.encoder_channels[-1])
        flatten_dim = self.encoded_channels * self.encoded_size * self.encoded_size
        
        self.fc_mu = nn.Sequential(
            nn.Linear(flatten_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(flatten_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        
        # ============ 解码器 ============
        self._build_decoder()
        
        # ============ 分类头 (用于评估准确率) ============
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
    
    def _build_encoder(self):
        """构建编码器"""
        layers = []
        in_channels = self.img_channels + self.num_classes  # 条件拼接
        
        for i, out_channels in enumerate(self.encoder_channels):
            # 下采样卷积
            layers.append(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            
            # 残差块
            for _ in range(self.num_res_blocks):
                layers.append(ResidualBlock(out_channels, self.use_bn))
            
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
    
    def _build_decoder(self):
        """构建解码器"""
        # 从潜在空间映射到特征图
        flatten_dim = self.encoded_channels * self.encoded_size * self.encoded_size
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, flatten_dim),
            nn.ReLU()
        )
        
        # 转置卷积上采样
        layers = []
        decoder_channels = self.encoder_channels[::-1]  # 反转通道列表
        
        for i, out_channels in enumerate(decoder_channels[1:] + [self.img_channels]):
            in_channels = decoder_channels[i]
            
            # 残差块
            for _ in range(self.num_res_blocks):
                layers.append(ResidualBlock(in_channels, self.use_bn))
            
            # 上采样
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
            
            if out_channels != self.img_channels:
                if self.use_bn:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def encode(self, x, c):
        """编码器: 输入图像和条件,输出潜在分布参数"""
        batch_size = x.size(0)
        
        # 将条件转换为one-hot并扩展为图像形式
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        c_map = c_onehot.view(batch_size, self.num_classes, 1, 1)
        c_map = c_map.expand(-1, -1, self.img_size, self.img_size)
        
        # 拼接输入和条件
        x_c = torch.cat([x, c_map], dim=1)
        
        # 编码
        h = self.encoder(x_c)
        h = h.view(batch_size, -1)
        
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """解码器: 从潜在变量和条件重构图像"""
        batch_size = z.size(0)
        
        # 条件one-hot
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        
        # 拼接潜在向量和条件
        z_c = torch.cat([z, c_onehot], dim=1)
        
        # 映射到特征图尺寸
        h = self.decoder_input(z_c)
        h = h.view(batch_size, self.encoded_channels, self.encoded_size, self.encoded_size)
        
        # 解码
        return self.decoder(h)
    
    def forward(self, x, c):
        """前向传播"""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar, z
    
    def classify(self, z):
        """从潜在表示进行分类"""
        return self.classifier(z)
    
    def sample(self, num_samples, c, device='cpu'):
        """条件采样"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        c = c.to(device)
        return self.decode(z, c)
    
    def get_latent(self, x, c):
        """获取潜在表示"""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class StudentCVAE(nn.Module):
    """
    学生CVAE模型 (用于知识蒸馏)
    比教师模型更小，但保持相同的潜在维度
    """
    
    # 学生模型配置 (比对应的教师模型小约50%)
    STUDENT_CONFIGS = {
        'small': {
            'encoder_channels': [32, 64, 128, 256],
            'latent_dim': 256,  # 保持与教师一致
            'hidden_dim': 512,
            'num_res_blocks': 1,
        },
        'medium': {
            'encoder_channels': [64, 128, 256, 512],
            'latent_dim': 512,
            'hidden_dim': 1024,
            'num_res_blocks': 1,
        },
        'large': {
            'encoder_channels': [128, 256, 512, 1024],
            'latent_dim': 1024,
            'hidden_dim': 2048,
            'num_res_blocks': 2,
        }
    }
    
    def __init__(self, img_size=32, img_channels=3, num_classes=10,
                 model_size='small', use_bn=True):
        super().__init__()
        
        assert model_size in self.STUDENT_CONFIGS
        config = self.STUDENT_CONFIGS[model_size]
        
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.model_size = model_size
        self.use_bn = use_bn
        
        self.encoder_channels = config['encoder_channels']
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_res_blocks = config['num_res_blocks']
        
        self.encoded_size = img_size // (2 ** len(self.encoder_channels))
        self.encoded_channels = self.encoder_channels[-1]
        
        # 编码器 (简化版)
        self._build_encoder()
        
        # 潜在空间
        flatten_dim = self.encoded_channels * self.encoded_size * self.encoded_size
        self.fc_mu = nn.Linear(flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim, self.latent_dim)
        
        # 解码器
        self._build_decoder()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, num_classes)
        )
    
    def _build_encoder(self):
        """构建简化的编码器"""
        layers = []
        in_channels = self.img_channels + self.num_classes
        
        for out_channels in self.encoder_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
    
    def _build_decoder(self):
        """构建简化的解码器"""
        flatten_dim = self.encoded_channels * self.encoded_size * self.encoded_size
        self.decoder_input = nn.Linear(self.latent_dim + self.num_classes, flatten_dim)
        
        layers = []
        decoder_channels = self.encoder_channels[::-1]
        
        for i, out_channels in enumerate(decoder_channels[1:] + [self.img_channels]):
            in_channels = decoder_channels[i]
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
            
            if out_channels != self.img_channels:
                if self.use_bn:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def encode(self, x, c):
        batch_size = x.size(0)
        c_onehot = F.one_hot(c, num_classes=self.num_classes).float()
        c_map = c_onehot.view(batch_size, self.num_classes, 1, 1)
        c_map = c_map.expand(-1, -1, self.img_size, self.img_size)
        
        x_c = torch.cat([x, c_map], dim=1)
        h = self.encoder(x_c)
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
        h = F.relu(h)
        h = h.view(batch_size, self.encoded_channels, self.encoded_size, self.encoded_size)
        
        return self.decoder(h)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar, z
    
    def classify(self, z):
        return self.classifier(z)


def cvae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    CVAE损失函数 = 重构损失 + KL散度
    """
    # 重构损失 (MSE for color images)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """计算模型大小(MB)"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def get_model_info():
    """获取所有模型配置的参数量信息"""
    info = {}
    for size in ['small', 'medium', 'large']:
        teacher = LargeCVAE(img_size=32, img_channels=3, num_classes=10, model_size=size)
        student = StudentCVAE(img_size=32, img_channels=3, num_classes=10, model_size=size)
        
        info[size] = {
            'teacher': {
                'params': count_parameters(teacher),
                'size_mb': get_model_size_mb(teacher)
            },
            'student': {
                'params': count_parameters(student),
                'size_mb': get_model_size_mb(student)
            },
            'compression_ratio': count_parameters(teacher) / count_parameters(student)
        }
    
    return info


if __name__ == "__main__":
    print("=" * 60)
    print("大规模CVAE模型测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试不同规模
    for model_size in ['small', 'medium', 'large']:
        print(f"\n{'='*20} {model_size.upper()} 模型 {'='*20}")
        
        # 教师模型
        teacher = LargeCVAE(
            img_size=32, img_channels=3, num_classes=10,
            model_size=model_size
        ).to(device)
        
        # 学生模型
        student = StudentCVAE(
            img_size=32, img_channels=3, num_classes=10,
            model_size=model_size
        ).to(device)
        
        teacher_params = count_parameters(teacher)
        student_params = count_parameters(student)
        
        print(f"\n教师模型:")
        print(f"  参数量: {teacher_params:,} ({teacher_params/1e6:.2f}M)")
        print(f"  模型大小: {get_model_size_mb(teacher):.2f} MB")
        
        print(f"\n学生模型:")
        print(f"  参数量: {student_params:,} ({student_params/1e6:.2f}M)")
        print(f"  模型大小: {get_model_size_mb(student):.2f} MB")
        print(f"  压缩比: {teacher_params/student_params:.2f}x")
        
        # 测试前向传播
        batch_size = 4
        x = torch.rand(batch_size, 3, 32, 32).to(device)
        c = torch.randint(0, 10, (batch_size,)).to(device)
        
        # 教师前向传播
        recon, mu, logvar, z = teacher(x, c)
        loss, recon_loss, kl_loss = cvae_loss_function(recon, x, mu, logvar)
        logits = teacher.classify(z)
        
        print(f"\n教师模型测试:")
        print(f"  输入: {x.shape}")
        print(f"  重构: {recon.shape}")
        print(f"  潜在: {z.shape}")
        print(f"  分类: {logits.shape}")
        print(f"  总损失: {loss.item()/batch_size:.2f}")
        
        # 学生前向传播
        recon_s, mu_s, logvar_s, z_s = student(x, c)
        logits_s = student.classify(z_s)
        
        print(f"\n学生模型测试:")
        print(f"  重构: {recon_s.shape}")
        print(f"  潜在: {z_s.shape}")
        print(f"  分类: {logits_s.shape}")
    
    print("\n" + "=" * 60)
    print("模型参数量汇总:")
    print("=" * 60)
    info = get_model_info()
    for size, data in info.items():
        print(f"\n{size.upper()}:")
        print(f"  教师: {data['teacher']['params']/1e6:.2f}M 参数, {data['teacher']['size_mb']:.2f} MB")
        print(f"  学生: {data['student']['params']/1e6:.2f}M 参数, {data['student']['size_mb']:.2f} MB")
        print(f"  压缩比: {data['compression_ratio']:.2f}x")

