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

def scan_and_copy_files():
    """扫描 /kaggle/input/ 并复制文件到 /kaggle/working/"""
    import shutil
    
    input_dir = '/kaggle/input'
    working_dir = '/kaggle/working'
    
    if not os.path.exists(input_dir):
        print("⚠️  /kaggle/input/ 目录不存在，跳过文件扫描")
        return
    
    print("📂 扫描 /kaggle/input/ 目录...")
    
    copied_pdfs = []
    copied_images = []
    
    # 递归扫描所有文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 跳过无效文件名
            if not file or file.startswith('.') or len(file) < 5:
                continue
            
            # 调试：显示所有文件
            print(f"   🔍 扫描到: {file}")
                
            src = os.path.join(root, file)
            dst = os.path.join(working_dir, file)
            
            try:
                # 修复：使用小写比较，支持 .pdf, .PDF, .Pdf 等
                if file.lower().endswith('.pdf'):
                    shutil.copy(src, dst)
                    copied_pdfs.append(file)
                    print(f"   ✅ 复制 PDF: {file}")
                elif any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
                    shutil.copy(src, dst)
                    copied_images.append(file)
                    print(f"   ✅ 复制图片: {file}")
                else:
                    print(f"   ⚪ 跳过非目标文件: {file}")
            except Exception as e:
                print(f"   ⚠️  复制文件失败 {file}: {e}")
    
    if copied_pdfs or copied_images:
        print(f"\n📁 复制完成: {len(copied_pdfs)} 个 PDF, {len(copied_images)} 张图片")
    else:
        print("⚠️  未找到 PDF 或图片文件")
        print("\n🔍 请检查:")
        print("   1. 文件是否已上传到 Kaggle")
        print("   2. 文件是否在 /kaggle/input/ 目录下")
        print("   3. 文件扩展名是否正确 (.pdf, .jpg, .png 等)")

def main():
    """主函数"""
    print("🚀 Kaggle简化多模态测试")
    print("="*50)
    
    # 设置环境
    setup_kaggle_environment()
    
    # 从 /kaggle/input/ 复制文件到 /kaggle/working/
    scan_and_copy_files()
    
    # 检查文件
    working_dir = '/kaggle/working'
    
    # 过滤有效的PDF文件（排除空文件名和隐藏文件）
    try:
        all_files = os.listdir(working_dir)
        
        # 修复：使用小写比较，支持 .pdf, .PDF, .Pdf 等
        pdf_files = [
            f for f in all_files 
            if f.lower().endswith('.pdf')  # 改为小写比较
            and len(f) > 4  # 确保不只是 '.pdf'
            and not f.startswith('.')  # 排除隐藏文件
            and os.path.isfile(os.path.join(working_dir, f))  # 确保是文件
        ]
        image_files = [
            f for f in all_files 
            if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])
            and not f.startswith('.')  # 排除隐藏文件
            and os.path.isfile(os.path.join(working_dir, f))  # 确保是文件
        ]
    except Exception as e:
        print(f"❌ 扫描文件时出错: {e}")
        pdf_files = []
        image_files = []
    
    print(f"\n📁 /kaggle/working/ 中的文件:")
    print(f"   - PDF文件: {len(pdf_files)} 个")
    for pdf in pdf_files:
        pdf_path = os.path.join(working_dir, pdf)
        file_size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        print(f"     * {pdf} ({file_size/1024:.1f} KB)")
    
    print(f"   - 图片文件: {len(image_files)} 个")
    for img in image_files:
        img_path = os.path.join(working_dir, img)
        file_size = os.path.getsize(img_path) if os.path.exists(img_path) else 0
        print(f"     * {img} ({file_size/1024:.1f} KB)")
    
    if not pdf_files and not image_files:
        print("\n💡 使用说明:")
        print("   1. 在 Kaggle Notebook 右侧点击 '+ Add data'")
        print("   2. 选择 'Upload' 标签")
        print("   3. 上传你的 PDF 和图片文件")
        print("   4. 重新运行此脚本")
        print("\n🔍 当前目录内容:")
        try:
            print(f"   {os.listdir(working_dir)}")
        except:
            pass
        return
    
    # 处理文件（添加路径验证）
    if pdf_files:
        pdf_path = os.path.join(working_dir, pdf_files[0])
        if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
            print(f"❌ PDF 文件路径无效: {pdf_path}")
            pdf_path = None
    else:
        pdf_path = None
    
    if image_files:
        image_paths = []
        for img in image_files:
            img_path = os.path.join(working_dir, img)
            if os.path.exists(img_path) and os.path.isfile(img_path):
                image_paths.append(img_path)
        image_paths = image_paths if image_paths else None
    else:
        image_paths = None
    
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