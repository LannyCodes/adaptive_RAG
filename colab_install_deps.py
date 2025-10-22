#!/usr/bin/env python3
"""
Colab环境依赖安装脚本
确保所有LangChain相关包都是最新版本，避免导入错误
"""

import subprocess
import sys

def install_package(package):
    """安装单个包"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

def main():
    print("="*70)
    print("📦 Colab GraphRAG 依赖安装")
    print("="*70)
    
    # 关键包列表（指定版本以确保兼容性）
    packages = [
        # LangChain核心包（最新版本）
        "langchain>=0.1.0",
        "langchain-core>=0.1.52",
        "langchain-community>=0.0.38",
        "langchain-text-splitters>=0.0.1",
        "langgraph>=0.0.40",
        
        # Ollama支持
        "langchain-ollama>=0.1.0",
        
        # 向量数据库和嵌入
        "chromadb>=0.4.22",
        "sentence-transformers>=2.2.0",
        
        # 文档处理
        "tiktoken>=0.5.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        
        # 网络搜索
        "tavily-python>=0.3.0",
        
        # 工具库
        "python-dotenv>=1.0.0",
        
        # GraphRAG特定
        "networkx>=3.1",
        "python-louvain>=0.16",
        
        # PyTorch和Transformers
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ]
    
    print("\n🔄 开始安装依赖包...\n")
    
    for i, package in enumerate(packages, 1):
        try:
            print(f"[{i}/{len(packages)}] 安装 {package}...")
            install_package(package)
            print(f"    ✅ {package} 安装成功")
        except Exception as e:
            print(f"    ❌ {package} 安装失败: {e}")
    
    print("\n" + "="*70)
    print("✅ 依赖安装完成!")
    print("="*70)
    
    # 验证关键导入
    print("\n🔍 验证关键导入...")
    
    imports_to_check = [
        ("langchain", "LangChain"),
        ("langchain_core", "LangChain Core"),
        ("langchain_community", "LangChain Community"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("networkx", "NetworkX"),
    ]
    
    all_ok = True
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            all_ok = False
    
    if all_ok:
        print("\n🎉 所有依赖验证通过!")
    else:
        print("\n⚠️  部分依赖验证失败，请检查错误信息")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
