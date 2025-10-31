"""
VAE训练过程详细示例
展示VAE在MNIST数据集上的完整训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from vae_model_structure import VAE, VAELoss


class VAEVisualizer:
    """VAE可视化工具类"""
    
    @staticmethod
    def plot_reconstruction(original, reconstructed, epoch):
        """绘制原始图像和重建图像的对比"""
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        
        for i in range(10):
            # 原始图像
            axes[0, i].imshow(original[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # 重建图像
            axes[1, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Epoch {epoch} - Reconstruction Comparison')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_latent_space(model, test_loader, device):
        """可视化潜在空间"""
        model.eval()
        
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                mu, _ = model.encoder(data.view(-1, 784))
                latent_vectors.append(mu.cpu().numpy())
                labels.append(target.numpy())
        
        latent_vectors = np.concatenate(latent_vectors)
        labels = np.concatenate(labels)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                             c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('2D Latent Space Visualization')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()


class VAETrainer:
    """VAE训练器类"""
    
    def __init__(self, model, train_loader, test_loader, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 记录训练历史
        self.train_losses = []
        self.test_losses = []
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # 前向传播
            recon_batch, mu, logvar = self.model(data.view(-1, 784))
            
            # 计算损失
            loss = VAELoss.loss_function(recon_batch, data.view(-1, 784), mu, logvar)
            
            # 反向传播
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
        
        avg_loss = train_loss / len(self.train_loader.dataset)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def test_epoch(self, epoch):
        """测试一个epoch"""
        self.model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data.view(-1, 784))
                test_loss += VAELoss.loss_function(recon_batch, data.view(-1, 784), mu, logvar).item()
        
        avg_loss = test_loss / len(self.test_loader.dataset)
        self.test_losses.append(avg_loss)
        
        print(f'====> Test set loss: {avg_loss:.4f}')
        return avg_loss
    
    def train(self, epochs=10):
        """完整训练过程"""
        print("🚀 开始VAE训练...")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            test_loss = self.test_epoch(epoch)
            
            # 每5个epoch可视化一次
            if epoch % 5 == 0:
                self.visualize_reconstruction(epoch)
        
        print("✅ 训练完成！")
        self.plot_training_history()
    
    def visualize_reconstruction(self, epoch):
        """可视化重建结果"""
        self.model.eval()
        
        with torch.no_grad():
            # 获取一批测试数据
            data_iter = iter(self.test_loader)
            test_data, _ = next(data_iter)
            test_data = test_data.to(self.device)
            
            # 重建
            recon_batch, _, _ = self.model(test_data.view(-1, 784))
            
            # 可视化
            VAEVisualizer.plot_reconstruction(test_data.view(-1, 784)[:10], 
                                             recon_batch[:10], epoch)
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Training History')
        plt.legend()
        plt.grid(True)
        plt.show()


# ============================================================================
# 数据准备
# ============================================================================

def load_mnist_data(batch_size=128):
    """加载MNIST数据集"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主训练函数"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("📦 加载MNIST数据集...")
    train_loader, test_loader = load_mnist_data()
    
    # 创建VAE模型
    print("🏗️  创建VAE模型...")
    model = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=20)
    
    # 创建训练器
    trainer = VAETrainer(model, train_loader, test_loader, device)
    
    # 开始训练
    trainer.train(epochs=10)
    
    # 可视化潜在空间
    print("\n🌌 可视化潜在空间...")
    VAEVisualizer.plot_latent_space(model, test_loader, device)
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = main()