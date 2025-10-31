import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """变分自编码器"""
    
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        
        # ============================================
        # Encoder (编码器)
        # ============================================
        
        # 卷积层 1: 1→32 channels, 28×28→14×14
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        # 卷积层 2: 32→64 channels, 14×14→7×7
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        # 全连接层: 3136→256
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        
        # 潜在空间分支
        self.fc_mu = nn.Linear(256, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(256, latent_dim)  # 对数方差
        
        # ============================================
        # Decoder (解码器)
        # ============================================
        
        # 全连接层: 20→256→3136
        self.fc2 = nn.Linear(latent_dim, 256)
        self.fc3 = nn.Linear(256, 64 * 7 * 7)
        
        # 转置卷积 1: 64→32 channels, 7×7→14×14
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        # 转置卷积 2: 32→1 channels, 14×14→28×28
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )
    
    def encode(self, x):
        """编码器: x → μ, log(σ²)"""
        # x: (batch, 1, 28, 28)
        
        h = F.relu(self.conv1(x))      # → (batch, 32, 14, 14)
        h = F.relu(self.conv2(h))      # → (batch, 64, 7, 7)
        h = h.view(-1, 64 * 7 * 7)     # → (batch, 3136)
        h = F.relu(self.fc1(h))        # → (batch, 256)
        
        mu = self.fc_mu(h)             # → (batch, 20)
        logvar = self.fc_logvar(h)     # → (batch, 20)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化: z = μ + σε"""
        std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2)
        eps = torch.randn_like(std)    # ε ~ N(0,1)
        z = mu + eps * std             # z = μ + σε
        return z
    
    def decode(self, z):
        """解码器: z → x'"""
        # z: (batch, 20)
        
        h = F.relu(self.fc2(z))        # → (batch, 256)
        h = F.relu(self.fc3(h))        # → (batch, 3136)
        h = h.view(-1, 64, 7, 7)       # → (batch, 64, 7, 7)
        h = F.relu(self.deconv1(h))    # → (batch, 32, 14, 14)
        x_recon = torch.sigmoid(self.deconv2(h))  # → (batch, 1, 28, 28)
        
        return x_recon
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)        # 编码
        z = self.reparameterize(mu, logvar) # 采样
        x_recon = self.decode(z)           # 解码
        return x_recon, mu, logvar


# ============================================
# 损失函数
# ============================================

def vae_loss(x_recon, x, mu, logvar):
    """
    VAE 损失 = 重建损失 + KL 散度
    
    Args:
        x_recon: 重建图像 (batch, 1, 28, 28)
        x: 原始图像 (batch, 1, 28, 28)
        mu: 均值 (batch, latent_dim)
        logvar: 对数方差 (batch, latent_dim)
    """
    # 1. 重建损失 (Binary Cross Entropy)
    # 衡量重建图像与原图的差异
    recon_loss = F.binary_cross_entropy(
        x_recon, x, reduction='sum'
    )
    
    # 2. KL 散度 (Kullback-Leibler Divergence)
    # 衡量 q(z|x) 与先验 p(z)=N(0,1) 的差异
    # KL(q||p) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    
    # 总损失
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================
# 使用示例
# ============================================

# 创建模型
model = VAE(latent_dim=20)

# 输入图像 (batch_size=32, channels=1, height=28, width=28)
x = torch.randn(32, 1, 28, 28)

# 前向传播
x_recon, mu, logvar = model(x)

# 计算损失
loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

print(f"重建形状: {x_recon.shape}")  # (32, 1, 28, 28)
print(f"μ 形状: {mu.shape}")         # (32, 20)
print(f"log(σ²) 形状: {logvar.shape}")  # (32, 20)
print(f"总损失: {loss.item():.2f}")
print(f"重建损失: {recon_loss.item():.2f}")
print(f"KL散度: {kl_loss.item():.2f}")