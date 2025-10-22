#!/usr/bin/env python3
"""
Google Colab GPU检测和GraphRAG性能测试脚本
可以直接在Colab中运行：python colab_gpu_test.py
"""

import sys
import time
import torch
import numpy as np
from typing import List, Dict

def print_section(title: str):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60 + "\n")


def test_gpu_availability():
    """测试GPU可用性"""
    print_section("🔍 GPU环境检测")
    
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   当前GPU: {torch.cuda.current_device()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   总显存: {total_memory:.2f} GB")
        
        return True
    else:
        print("\n⚠️  警告: 未检测到GPU")
        print("   在Colab中启用GPU: 运行时 → 更改运行时类型 → GPU")
        return False


def benchmark_matrix_multiplication(matrix_size=5000):
    """GPU vs CPU 矩阵运算性能测试"""
    print_section("⚡ GPU vs CPU 矩阵运算性能测试")
    
    print(f"矩阵大小: {matrix_size}x{matrix_size}\n")
    
    # CPU测试
    print("🔵 CPU测试...")
    a_cpu = torch.randn(matrix_size, matrix_size)
    b_cpu = torch.randn(matrix_size, matrix_size)
    
    start = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"   CPU时间: {cpu_time:.2f} 秒")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n🟢 GPU测试...")
        a_gpu = torch.randn(matrix_size, matrix_size).cuda()
        b_gpu = torch.randn(matrix_size, matrix_size).cuda()
        
        # 预热GPU
        _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"   GPU时间: {gpu_time:.2f} 秒")
        
        speedup = cpu_time / gpu_time
        print(f"\n🚀 加速比: {speedup:.2f}x")
        print(f"   GPU比CPU快 {speedup:.1f} 倍!")
        
        return speedup
    else:
        print("\n⚠️  跳过GPU测试（GPU不可用）")
        return 1.0


def test_text_embedding_performance():
    """测试文本嵌入性能（需要sentence-transformers）"""
    print_section("📝 文本嵌入性能测试")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 准备测试数据
        test_texts = [
            "Large Language Models are transforming AI",
            "GraphRAG combines knowledge graphs with retrieval",
            "GPU acceleration significantly improves performance",
            "Natural language processing is advancing rapidly",
        ] * 250  # 1000个文本
        
        print(f"测试数据: {len(test_texts)} 个文本\n")
        
        # CPU测试
        print("🔵 CPU嵌入测试...")
        model_cpu = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device='cpu'
        )
        start = time.time()
        embeddings_cpu = model_cpu.encode(test_texts, show_progress_bar=False, batch_size=32)
        cpu_time = time.time() - start
        print(f"   CPU时间: {cpu_time:.2f}秒")
        print(f"   速度: {len(test_texts)/cpu_time:.1f} 文本/秒")
        
        # GPU测试
        if torch.cuda.is_available():
            print("\n🟢 GPU嵌入测试...")
            model_gpu = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cuda'
            )
            start = time.time()
            embeddings_gpu = model_gpu.encode(test_texts, show_progress_bar=False, batch_size=32)
            gpu_time = time.time() - start
            print(f"   GPU时间: {gpu_time:.2f}秒")
            print(f"   速度: {len(test_texts)/gpu_time:.1f} 文本/秒")
            
            speedup = cpu_time / gpu_time
            print(f"\n🚀 加速比: {speedup:.2f}x")
            print(f"   节省时间: {cpu_time - gpu_time:.2f}秒")
            
            return speedup
        else:
            print("\n⚠️  跳过GPU测试")
            return 1.0
            
    except ImportError:
        print("⚠️  sentence-transformers未安装")
        print("   安装: pip install sentence-transformers")
        return None


def monitor_gpu_memory():
    """监控GPU显存使用"""
    if not torch.cuda.is_available():
        return
    
    print_section("💾 GPU显存使用情况")
    
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"已分配: {allocated:.2f} GB")
    print(f"已保留: {reserved:.2f} GB")
    print(f"总显存: {total:.2f} GB")
    print(f"使用率: {(allocated/total)*100:.1f}%")


def generate_performance_report(matrix_speedup, embedding_speedup):
    """生成性能报告"""
    print_section("📈 性能测试总结报告")
    
    print("🖥️  硬件信息:")
    if torch.cuda.is_available():
        print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        print("   ⚠️  GPU不可用")
    
    print(f"\n   PyTorch版本: {torch.__version__}")
    print(f"   Python版本: {sys.version.split()[0]}")
    
    print("\n⚡ 性能测试结果:")
    print(f"   矩阵运算加速: {matrix_speedup:.2f}x")
    if embedding_speedup:
        print(f"   文本嵌入加速: {embedding_speedup:.2f}x")
    
    print("\n💡 建议:")
    if torch.cuda.is_available():
        print("   ✅ GPU运行良好！")
        print("   ✅ 建议在Colab上运行完整的GraphRAG索引构建")
        print("   ✅ 预计索引构建时间将缩短 3-5 倍")
        
        # 估算时间节省
        if embedding_speedup and embedding_speedup > 1:
            print(f"\n⏱️  时间节省估算:")
            print(f"   100文档CPU耗时: ~15分钟")
            print(f"   100文档GPU耗时: ~{15/embedding_speedup:.1f}分钟")
            print(f"   节省: ~{15 - 15/embedding_speedup:.1f}分钟")
    else:
        print("   ⚠️  建议启用GPU以获得最佳性能")
        print("   ⚠️  Colab启用GPU: 运行时 → 更改运行时类型 → GPU")


def install_dependencies():
    """安装必要的依赖（仅在Colab中）"""
    try:
        import google.colab
        is_colab = True
    except:
        is_colab = False
    
    if is_colab:
        print_section("📦 安装依赖")
        print("检测到Colab环境，安装必要的包...\n")
        
        import subprocess
        packages = [
            'sentence-transformers',
            'networkx',
            'python-louvain',
        ]
        
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} 已安装")
            except ImportError:
                print(f"📥 安装 {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
                print(f"✅ {package} 安装完成")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 Google Colab GPU检测和GraphRAG性能测试")
    print("="*60)
    
    # 检查是否在Colab中运行
    try:
        import google.colab
        print("\n✅ 运行环境: Google Colab")
    except:
        print("\n⚠️  警告: 未检测到Colab环境")
        print("   本脚本专为Google Colab设计")
    
    # 安装依赖
    install_dependencies()
    
    # 1. GPU检测
    gpu_available = test_gpu_availability()
    
    # 2. 矩阵运算性能测试
    matrix_speedup = benchmark_matrix_multiplication(matrix_size=5000)
    
    # 3. 文本嵌入性能测试
    embedding_speedup = test_text_embedding_performance()
    
    # 4. 显存监控
    if gpu_available:
        monitor_gpu_memory()
    
    # 5. 生成报告
    generate_performance_report(matrix_speedup, embedding_speedup)
    
    print("\n" + "="*60)
    print("✅ 测试完成!")
    print("="*60)
    
    print("\n📚 下一步:")
    print("   1. 如果GPU测试成功，可以上传完整的adaptive_RAG项目")
    print("   2. 运行 main_graphrag.py 进行完整的知识图谱构建")
    print("   3. 享受GPU带来的3-5倍速度提升!")


if __name__ == "__main__":
    main()
