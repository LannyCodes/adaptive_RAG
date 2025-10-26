#!/usr/bin/env python3
"""
Kaggle Ollama 连接问题诊断和修复脚本
解决 GraphRAG 异步处理时的连接错误
"""

import subprocess
import time
import requests
import os

def check_ollama_service():
    """检查 Ollama 服务状态"""
    print("="*70)
    print("🔍 Ollama 服务诊断")
    print("="*70)
    
    # 1. 检查进程
    print("\n1️⃣ 检查 Ollama 进程...")
    ps_check = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True)
    
    if ps_check.returncode == 0:
        print("   ✅ Ollama 进程正在运行")
        pids = ps_check.stdout.decode().strip().split('\n')
        print(f"   📊 进程 PID: {', '.join(pids)}")
    else:
        print("   ❌ Ollama 进程未运行")
        return False
    
    # 2. 检查端口
    print("\n2️⃣ 检查端口 11434...")
    port_check = subprocess.run(
        ['netstat', '-tuln'], 
        capture_output=True, 
        text=True
    )
    
    if '11434' in port_check.stdout:
        print("   ✅ 端口 11434 已监听")
    else:
        print("   ❌ 端口 11434 未监听")
        return False
    
    # 3. 测试 API 连接
    print("\n3️⃣ 测试 API 连接...")
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("   ✅ API 连接正常")
            models = response.json().get('models', [])
            print(f"   📦 可用模型: {len(models)}")
            for model in models:
                print(f"      • {model.get('name', 'unknown')}")
            return True
        else:
            print(f"   ❌ API 返回错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API 连接失败: {e}")
        return False

def start_ollama_service():
    """启动 Ollama 服务"""
    print("\n"+"="*70)
    print("🚀 启动 Ollama 服务")
    print("="*70)
    
    # 先杀死可能存在的僵尸进程
    print("\n1️⃣ 清理旧进程...")
    subprocess.run(['pkill', '-9', 'ollama'], capture_output=True)
    time.sleep(2)
    
    # 启动服务
    print("\n2️⃣ 启动新服务...")
    process = subprocess.Popen(
        ['ollama', 'serve'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy()
    )
    
    print(f"   ✅ 服务进程已启动 (PID: {process.pid})")
    
    # 等待服务就绪
    print("\n3️⃣ 等待服务就绪...")
    max_wait = 30
    for i in range(max_wait):
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                print(f"   ✅ 服务就绪！(耗时 {i+1} 秒)")
                return True
        except:
            pass
        
        if i < max_wait - 1:
            print(f"   ⏳ 等待中... ({i+1}/{max_wait})", end='\r')
            time.sleep(1)
    
    print(f"\n   ⚠️ 服务启动超时，但可能仍在初始化中")
    return False

def test_generation():
    """测试生成功能"""
    print("\n"+"="*70)
    print("🧪 测试文本生成")
    print("="*70)
    
    print("\n   ℹ️ 首次调用会加载模型到内存，需要 30-60 秒...")
    print("   ⏳ 请耐心等待...\n")
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "mistral",
                "prompt": "Say 'Hello' in one word",
                "stream": False
            },
            timeout=120  # 增加到 120 秒，首次加载模型需要时间
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 生成成功")
            print(f"   📝 响应: {result.get('response', '')[:100]}")
            return True
        else:
            print(f"   ❌ 生成失败: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"   ⚠️ 生成超时（但这可能是模型加载中）")
        print(f"   💡 建议：再等待 30 秒后重试")
        return False
    except Exception as e:
        print(f"   ❌ 生成错误: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "="*70)
    print("🔧 Kaggle Ollama 连接问题修复工具")
    print("="*70)
    print("\n解决问题: Cannot connect to host localhost:11434")
    print("场景: GraphRAG 异步批处理时")
    
    # 检查服务
    is_running = check_ollama_service()
    
    if not is_running:
        print("\n⚠️ Ollama 服务未正常运行，正在修复...")
        start_ollama_service()
        
        # 再次检查
        print("\n"+"="*70)
        print("🔍 验证修复结果")
        print("="*70)
        is_running = check_ollama_service()
    
    # 测试生成
    if is_running:
        test_generation()
    
    # 输出建议
    print("\n"+"="*70)
    print("💡 使用建议")
    print("="*70)
    
    if is_running:
        if test_generation():
            print("""
✅ Ollama 服务完全就绪！现在可以运行 GraphRAG 了

📝 在 Kaggle Notebook 中运行:

from document_processor import DocumentProcessor
from graph_indexer import GraphRAGIndexer

# 初始化
processor = DocumentProcessor()
vectorstore, retriever, doc_splits = processor.setup_knowledge_base(
    enable_graphrag=True
)

# GraphRAG 索引（异步处理）
indexer = GraphRAGIndexer(
    enable_async=True,      # 启用异步
    async_batch_size=8      # 并发处理 8 个文档
)

graph = indexer.index_documents(doc_splits)
        """)
        else:
            print("""
⚠️ Ollama 服务运行中，但模型可能还在加载

💡 解决方案：

1. 等待 30-60 秒让模型完全加载
2. 再次运行此脚本验证
3. 或者直接运行一次简单测试：
   !curl http://localhost:11434/api/generate -d '{
     "model": "mistral",
     "prompt": "Hello",
     "stream": false
   }'

4. 如果上述测试成功，就可以运行 GraphRAG 了
        """)
    else:
        print("""
❌ Ollama 服务仍然异常

🔧 手动修复步骤:

1. 在 Kaggle Notebook 新单元格运行:
   !pkill -9 ollama
   !ollama serve &
   
2. 等待 15 秒后，运行:
   !curl http://localhost:11434/api/tags
   
3. 如果成功，重新运行此脚本验证

4. 如果失败，检查 Ollama 是否正确安装:
   !which ollama
   !ollama --version
        """)
    
    print("="*70)

if __name__ == "__main__":
    main()
