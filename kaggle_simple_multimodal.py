"""
Kaggle简化多模态测试脚本
用于在Kaggle环境中直接处理已上传的PDF和图片文件
"""

import os
import sys
import subprocess
import time
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, '/kaggle/working/adaptive_RAG')

# 导入项目模块
from document_processor import DocumentProcessor
from main import AdaptiveRAGSystem
from config import ENABLE_MULTIMODAL, SUPPORTED_IMAGE_FORMATS

def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("🔧 设置Kaggle环境...")
    
    # 安装必要的依赖
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                   'PyPDF2', 'pdfplumber', 'Pillow'])
    
    print("✅ 环境设置完成")

def process_uploaded_files(pdf_path: str = None, image_paths: List[str] = None):
    """
    处理已上传的文件
    
    Args:
        pdf_path: PDF文件路径
        image_paths: 图片路径列表
    """
    # 初始化文档处理器
    print("🔧 正在初始化文档处理器...")
    doc_processor = DocumentProcessor()
    
    # 处理PDF文件
    if pdf_path and os.path.exists(pdf_path):
        print(f"📄 处理PDF文件: {pdf_path}")
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 分割文档
            doc_splits = doc_processor.split_documents(docs)
            
            # 创建向量数据库
            vectorstore, retriever = doc_processor.create_vectorstore(doc_splits)
            
            print(f"✅ PDF处理完成，共 {len(doc_splits)} 个文档块")
        except Exception as e:
            print(f"❌ PDF处理失败: {e}")
            return None
    else:
        # 使用默认知识库
        print("📄 使用默认知识库...")
        vectorstore, retriever, doc_splits = doc_processor.setup_knowledge_base()
    
    # 初始化RAG系统
    print("🤖 正在初始化自适应RAG系统...")
    rag_system = AdaptiveRAGSystem()
    
    # 更新RAG系统的检索器
    rag_system.retriever = retriever
    rag_system.doc_processor = doc_processor
    rag_system.workflow_nodes.retriever = retriever
    rag_system.workflow_nodes.doc_processor = doc_processor
    
    return rag_system, doc_processor

def query_with_multimodal(rag_system: AdaptiveRAGSystem, query: str, image_paths: List[str] = None):
    """
    执行多模态查询
    
    Args:
        rag_system: RAG系统实例
        query: 查询字符串
        image_paths: 图片路径列表
    """
    print(f"🔍 查询: {query}")
    
    try:
        # 执行查询
        result = rag_system.query(query)
        
        # 显示结果
        print("\n🎯 答案:")
        print(result['answer'])
        
        # 显示评估指标
        if result.get('retrieval_metrics'):
            metrics = result['retrieval_metrics']
            print("\n📊 检索评估:")
            print(f"   - 检索耗时: {metrics.get('latency', 0):.4f}秒")
            print(f"   - 检索文档数: {metrics.get('retrieved_docs_count', 0)}")
            print(f"   - Precision@3: {metrics.get('precision_at_3', 0):.4f}")
            print(f"   - Recall@3: {metrics.get('recall_at_3', 0):.4f}")
            print(f"   - MAP: {metrics.get('map_score', 0):.4f}")
        
        return result
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return None

def main():
    """主函数"""
    print("🚀 Kaggle简化多模态测试")
    print("="*50)
    
    # 设置环境
    setup_kaggle_environment()
    
    # 检查上传的文件
    working_dir = '/kaggle/working'
    pdf_files = [f for f in os.listdir(working_dir) if f.endswith('.pdf')]
    image_files = [f for f in os.listdir(working_dir) if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])]
    
    print(f"\n📁 发现文件:")
    print(f"   - PDF文件: {len(pdf_files)} 个")
    for pdf in pdf_files:
        print(f"     * {pdf}")
    
    print(f"   - 图片文件: {len(image_files)} 个")
    for img in image_files:
        print(f"     * {img}")
    
    # 处理文件
    pdf_path = os.path.join(working_dir, pdf_files[0]) if pdf_files else None
    image_paths = [os.path.join(working_dir, img) for img in image_files] if image_files else None
    
    rag_system, doc_processor = process_uploaded_files(pdf_path, image_paths)
    
    if not rag_system:
        print("❌ 系统初始化失败")
        return
    
    # 示例查询
    print("\n" + "="*50)
    print("🧪 示例查询测试")
    print("="*50)
    
    # 文本查询示例
    query1 = "请总结文档的主要内容"
    query_with_multimodal(rag_system, query1, image_paths)
    
    # 如果有图片，执行多模态查询
    if image_paths and ENABLE_MULTIMODAL:
        print("\n" + "="*50)
        print("🖼️ 多模态查询测试")
        print("="*50)
        
        query2 = "请结合图片内容，解释文档中的相关概念"
        query_with_multimodal(rag_system, query2, image_paths)
    
    print("\n" + "="*50)
    print("✅ 测试完成")
    print("="*50)
    print("\n💡 您可以继续使用以下代码进行自定义查询:")
    print("```python")
    print("# 自定义查询")
    print("custom_query = '您的问题'")
    print("query_with_multimodal(rag_system, custom_query, image_paths)")
    print("```")

if __name__ == "__main__":
    main()