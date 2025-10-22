#!/usr/bin/env python3
"""
Google Colab一键GPU测试脚本
复制此文件内容到Colab单元格中直接运行

使用方法:
1. 在Colab中创建新笔记本
2. 启用GPU (运行时 → 更改运行时类型 → GPU)
3. 复制并运行此脚本
"""

# ============================================================
# 🔧 自动安装依赖
# ============================================================
print("📦 检查并安装依赖...")
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# 检查必要的包
required_packages = {
    'torch': 'torch',
    'sentence_transformers': 'sentence-transformers',
    'networkx': 'networkx',
    'numpy': 'numpy'
}

for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"✅ {package_name} 已安装")
    except ImportError:
        print(f"📥 安装 {package_name}...")
        install(package_name)

print("\n" + "="*70)
print("🚀 Google Colab GPU性能测试 - GraphRAG加速验证")
print("="*70)

# ============================================================
# 1️⃣ GPU检测
# ============================================================
import torch
import time

print("\n" + "="*70)
print("🔍 步骤1: GPU环境检测")
print("="*70)

cuda_available = torch.cuda.is_available()
print(f"\n{'✅' if cuda_available else '❌'} CUDA可用: {cuda_available}")

if cuda_available:
    print(f"   📊 GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"   💾 显存大小: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"   🔢 CUDA版本: {torch.version.cuda}")
    print(f"   📈 PyTorch版本: {torch.__version__}")
else:
    print("\n⚠️  GPU未启用！")
    print("   请按照以下步骤启用GPU:")
    print("   1. 点击顶部菜单 '运行时'")
    print("   2. 选择 '更改运行时类型'")
    print("   3. 硬件加速器选择 'GPU'")
    print("   4. 点击 '保存'")
    print("   5. 重新运行此单元格")
    print("\n⚠️  测试将继续，但GPU相关测试会被跳过")

# ============================================================
# 2️⃣ 矩阵运算性能测试
# ============================================================
print("\n" + "="*70)
print("⚡ 步骤2: 矩阵运算性能测试")
print("="*70)

matrix_size = 5000
print(f"\n测试配置: {matrix_size}x{matrix_size} 矩阵乘法\n")

# CPU测试
print("🔵 CPU性能测试...")
a_cpu = torch.randn(matrix_size, matrix_size)
b_cpu = torch.randn(matrix_size, matrix_size)

start = time.time()
c_cpu = torch.mm(a_cpu, b_cpu)
cpu_time = time.time() - start

print(f"   ⏱️  CPU耗时: {cpu_time:.3f}秒")

# GPU测试
if cuda_available:
    print("\n🟢 GPU性能测试...")
    a_gpu = torch.randn(matrix_size, matrix_size).cuda()
    b_gpu = torch.randn(matrix_size, matrix_size).cuda()
    
    # 预热
    _ = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"   ⏱️  GPU耗时: {gpu_time:.3f}秒")
    
    speedup = cpu_time / gpu_time
    print(f"\n   🚀 性能提升: {speedup:.1f}x")
    print(f"   💡 GPU比CPU快 {speedup:.1f} 倍!")
    
    matrix_speedup = speedup
else:
    print("\n⚠️  跳过GPU测试")
    matrix_speedup = 1.0

# ============================================================
# 3️⃣ 文本嵌入性能测试
# ============================================================
print("\n" + "="*70)
print("📝 步骤3: 文本嵌入性能测试 (GraphRAG核心组件)")
print("="*70)

try:
    from sentence_transformers import SentenceTransformer
    
    # 准备测试数据
    test_texts = [
        "GraphRAG combines knowledge graphs with retrieval augmented generation",
        "GPU acceleration significantly improves machine learning performance",
        "Large language models benefit from efficient embedding computation",
        "Knowledge graph construction requires entity and relation extraction",
    ] * 250  # 1000条文本
    
    print(f"\n测试配置: {len(test_texts)}条文本嵌入\n")
    
    # CPU嵌入
    print("🔵 CPU嵌入测试...")
    model_cpu = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    
    start = time.time()
    embeddings_cpu = model_cpu.encode(test_texts, show_progress_bar=False, batch_size=32)
    cpu_emb_time = time.time() - start
    
    print(f"   ⏱️  CPU耗时: {cpu_emb_time:.2f}秒")
    print(f"   📊 处理速度: {len(test_texts)/cpu_emb_time:.1f} 文本/秒")
    
    # GPU嵌入
    if cuda_available:
        print("\n🟢 GPU嵌入测试...")
        model_gpu = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
        
        start = time.time()
        embeddings_gpu = model_gpu.encode(test_texts, show_progress_bar=False, batch_size=32)
        gpu_emb_time = time.time() - start
        
        print(f"   ⏱️  GPU耗时: {gpu_emb_time:.2f}秒")
        print(f"   📊 处理速度: {len(test_texts)/gpu_emb_time:.1f} 文本/秒")
        
        emb_speedup = cpu_emb_time / gpu_emb_time
        print(f"\n   🚀 性能提升: {emb_speedup:.1f}x")
        print(f"   ⏱️  节省时间: {cpu_emb_time - gpu_emb_time:.2f}秒")
    else:
        print("\n⚠️  跳过GPU测试")
        emb_speedup = 1.0
        
except ImportError:
    print("\n⚠️  sentence-transformers未安装，跳过此测试")
    emb_speedup = None

# ============================================================
# 4️⃣ GraphRAG场景模拟
# ============================================================
print("\n" + "="*70)
print("🔍 步骤4: GraphRAG实际场景模拟")
print("="*70)

if cuda_available and emb_speedup:
    print("\n模拟GraphRAG索引构建过程...\n")
    
    # 假设100个文档块的索引构建
    documents_count = 100
    
    # 实体提取时间 (每个文档约1秒)
    entity_extraction_time = documents_count * 1.0
    
    # 文本嵌入时间 (基于实际测试)
    # 假设每个文档平均产生10个实体，共1000个实体需要嵌入
    entities_count = documents_count * 10
    
    cpu_total_time = entity_extraction_time + (entities_count / (len(test_texts)/cpu_emb_time))
    gpu_total_time = entity_extraction_time + (entities_count / (len(test_texts)/gpu_emb_time))
    
    print(f"📊 场景: {documents_count}个文档的GraphRAG索引构建\n")
    print(f"🔵 CPU预计时间:")
    print(f"   - 实体提取: {entity_extraction_time/60:.1f}分钟")
    print(f"   - 向量嵌入: {(entities_count / (len(test_texts)/cpu_emb_time))/60:.1f}分钟")
    print(f"   - 总计: {cpu_total_time/60:.1f}分钟")
    
    print(f"\n🟢 GPU预计时间:")
    print(f"   - 实体提取: {entity_extraction_time/60:.1f}分钟 (相同)")
    print(f"   - 向量嵌入: {(entities_count / (len(test_texts)/gpu_emb_time))/60:.1f}分钟")
    print(f"   - 总计: {gpu_total_time/60:.1f}分钟")
    
    total_speedup = cpu_total_time / gpu_total_time
    time_saved = (cpu_total_time - gpu_total_time) / 60
    
    print(f"\n🚀 整体加速: {total_speedup:.1f}x")
    print(f"⏱️  节省时间: {time_saved:.1f}分钟")

# ============================================================
# 5️⃣ GPU显存监控
# ============================================================
if cuda_available:
    print("\n" + "="*70)
    print("💾 步骤5: GPU显存使用监控")
    print("="*70)
    
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\n   已分配: {allocated:.2f} GB")
    print(f"   已保留: {reserved:.2f} GB")
    print(f"   总显存: {total:.2f} GB")
    print(f"   使用率: {(allocated/total)*100:.1f}%")

# ============================================================
# 6️⃣ 性能总结
# ============================================================
print("\n" + "="*70)
print("📈 最终性能报告")
print("="*70)

print("\n🖥️  硬件配置:")
if cuda_available:
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
else:
    print("   ⚠️  GPU未启用")

print(f"\n⚡ 性能测试结果:")
print(f"   矩阵运算加速: {matrix_speedup:.1f}x")
if emb_speedup:
    print(f"   文本嵌入加速: {emb_speedup:.1f}x")
    if cuda_available:
        print(f"   GraphRAG整体加速: {total_speedup:.1f}x")

print("\n💡 结论和建议:")
if cuda_available:
    print("   ✅ GPU性能测试成功!")
    print("   ✅ 强烈建议在Colab GPU环境运行GraphRAG")
    print(f"   ✅ 预计可节省 {time_saved:.0f}+ 分钟的索引构建时间")
    print("\n📚 下一步:")
    print("   1. 上传adaptive_RAG项目文件到Colab")
    print("   2. 运行 main_graphrag.py 构建完整知识图谱")
    print("   3. 下载结果到本地使用")
else:
    print("   ⚠️  请启用GPU以获得最佳性能")
    print("   ⚠️  路径: 运行时 → 更改运行时类型 → GPU")

print("\n" + "="*70)
print("✅ 测试完成! 感谢使用GraphRAG GPU测试工具")
print("="*70)

# ============================================================
# 7️⃣ 可选: 显示nvidia-smi
# ============================================================
if cuda_available:
    print("\n📊 nvidia-smi 详细信息:")
    print("="*70)
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("⚠️  无法执行nvidia-smi命令")
