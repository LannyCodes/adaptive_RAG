"""
VAE（变分自编码器）模型完整结构解析
包含编码器、解码器、重参数化技巧和损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """编码器：将输入数据映射到潜在空间的均值和方差"""
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=20):
        super(Encoder, self).__init__()
        
        # 第1层：输入层 → 第一个隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        
        # 第2层：第一个隐藏层 → 第二个隐藏层
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # 第3层：第二个隐藏层 → 潜在空间均值
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        
        # 第4层：第二个隐藏层 → 潜在空间对数方差
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
    def forward(self, x):
        print("\n🔍 编码器前向传播过程：")
        print(f"输入形状: {x.shape}")
        
        # Layer 1: 输入 → 隐藏层1
        h1 = F.relu(self.fc1(x))
        print(f"Layer 1 后: {h1.shape}")
        
        # Layer 2: 隐藏层1 → 隐藏层2
        h2 = F.relu(self.fc2(h1))
        print(f"Layer 2 后: {h2.shape}")
        
        # Layer 3: 计算均值 μ
        mu = self.fc_mu(h2)
        print(f"均值 μ 形状: {mu.shape}")
        
        # Layer 4: 计算对数方差 log(σ²)
        logvar = self.fc_logvar(h2)
        print(f"对数方差 logvar 形状: {logvar.shape}")
        
        return mu, logvar


class Decoder(nn.Module):
    """解码器：从潜在空间重建原始数据"""
    
    def __init__(self, latent_dim=20, hidden_dims=[256, 512], output_dim=784):
        super(Decoder, self).__init__()
        
        # 第1层：潜在空间 → 第一个隐藏层
        self.fc1 = nn.Linear(latent_dim, hidden_dims[0])
        
        # 第2层：第一个隐藏层 → 第二个隐藏层
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # 第3层：第二个隐藏层 → 输出层
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        
    def forward(self, z):
        print("\n🔧 解码器前向传播过程：")
        print(f"潜在变量 z 形状: {z.shape}")
        
        # Layer 1: 潜在空间 → 隐藏层1
        h1 = F.relu(self.fc1(z))
        print(f"Layer 1 后: {h1.shape}")
        
        # Layer 2: 隐藏层1 → 隐藏层2
        h2 = F.relu(self.fc2(h1))
        print(f"Layer 2 后: {h2.shape}")
        
        # Layer 3: 隐藏层2 → 输出（使用sigmoid确保值在[0,1]）
        recon_x = torch.sigmoid(self.fc3(h2))
        print(f"重建输出形状: {recon_x.shape}")
        
        return recon_x


class Reparameterization:
    """重参数化技巧：从N(μ, σ²)采样，同时保持梯度可传播"""
    
    @staticmethod
    def reparameterize(mu, logvar):
        print("\n🔄 重参数化过程：")
        print(f"输入均值 μ: {mu.shape}, 对数方差 logvar: {logvar.shape}")
        
        # 计算标准差 σ = exp(0.5 * log(σ²))
        std = torch.exp(0.5 * logvar)
        print(f"标准差 σ 形状: {std.shape}")
        
        # 从标准正态分布采样 ε ~ N(0, I)
        eps = torch.randn_like(std)
        print(f"噪声 ε 形状: {eps.shape}")
        
        # 重参数化：z = μ + σ ⊙ ε
        z = mu + eps * std
        print(f"采样结果 z 形状: {z.shape}")
        
        return z


class VAELoss:
    """VAE损失函数：重建损失 + KL散度"""
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        print("\n📊 损失计算过程：")
        
        # 1. 重建损失（二元交叉熵）
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        print(f"重建损失 BCE: {BCE.item():.2f}")
        
        # 2. KL散度（潜在分布与标准正态分布的差异）
        # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        print(f"KL散度 KLD: {KLD.item():.2f}")
        
        # 3. 总损失
        total_loss = BCE + KLD
        print(f"总损失: {total_loss.item():.2f} (BCE + KLD)")
        
        return total_loss


class VAE(nn.Module):
    """完整的VAE模型"""
    
    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=20):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        
    def forward(self, x):
        print("=" * 60)
        print("🚀 VAE 完整前向传播")
        print("=" * 60)
        
        # 编码器：x → (μ, logvar)
        mu, logvar = self.encoder(x)
        
        # 重参数化：从N(μ, σ²)采样z
        z = Reparameterization.reparameterize(mu, logvar)
        
        # 解码器：z → 重建数据
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar


# ============================================================================
# 测试代码
# ============================================================================

def test_vae():
    """测试VAE模型结构"""
    
    print("🧪 开始测试VAE模型...")
    
    # 创建模拟数据（batch_size=4, 输入维度784）
    batch_size = 4
    input_dim = 784
    x = torch.randn(batch_size, input_dim)
    
    print(f"\n📦 输入数据形状: {x.shape}")
    
    # 初始化VAE模型
    model = VAE(input_dim=input_dim)
    
    # 前向传播
    recon_x, mu, logvar = model(x)
    
    # 计算损失
    loss = VAELoss.loss_function(recon_x, x, mu, logvar)
    
    print("\n✅ VAE模型测试完成！")
    
    return model, loss


if __name__ == "__main__":
    test_vae()