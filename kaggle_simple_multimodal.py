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
    处理已上传的文件，向量化并持久化到项目目录
    支持文件去重，避免重复处理
    
    Args:
        pdf_path: PDF文件路径
        image_paths: 图片路径列表
    """
    import hashlib
    import json
    
    # 设置向量数据库持久化目录（相对路径）
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(current_dir, 'chroma_db')
    metadata_file = os.path.join(current_dir, 'document_metadata.json')
    os.makedirs(persist_dir, exist_ok=True)
    
    print(f"💾 向量数据库持久化目录: {persist_dir}")
    
    # 加载已处理文件的元数据（用于去重）
    processed_files = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                processed_files = metadata.get('processed_files', {})
                print(f"📊 已加载元数据，发现 {len(processed_files)} 个已处理的文件")
        except Exception as e:
            print(f"⚠️  加载元数据失败: {e}")
    
    # 计算文件哈希值（用于去重检测）
    def get_file_hash(file_path: str) -> str:
        """计算文件的MD5哈希值"""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"⚠️  计算文件哈希失败: {e}")
            return None
    
    # 检查是否已存在向量数据库
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("✅ 检测到已存在的向量数据库，加载中...")
        try:
            # 加载已存在的向量数据库
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            from config import EMBEDDING_MODEL, COLLECTION_NAME
            
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            
            retriever = vectorstore.as_retriever()
            print(f"✅ 已加载持久化的向量数据库，共 {vectorstore._collection.count()} 个文档块")
            
            # 初始化文档处理器
            doc_processor = DocumentProcessor()
            
            # 检查PDF文件是否需要处理
            if pdf_path and os.path.exists(pdf_path):
                file_hash = get_file_hash(pdf_path)
                if file_hash and file_hash in processed_files:
                    print(f"⏭️  PDF文件已处理过（{pdf_path}），跳过")
                else:
                    print(f"🆕 检测到新PDF文件，正在添加: {pdf_path}")
                    try:
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(pdf_path)
                        docs = loader.load()
                        doc_splits = doc_processor.split_documents(docs)
                        
                        # 添加到现有向量数据库
                        vectorstore.add_documents(doc_splits)
                        print(f"✅ 已添加 {len(doc_splits)} 个新文档块")
                        
                        # 更新元数据
                        if file_hash:
                            processed_files[file_hash] = {
                                'path': pdf_path,
                                'type': 'pdf',
                                'chunks': len(doc_splits),
                                'processed_at': time.time()
                            }
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump({'processed_files': processed_files}, f, ensure_ascii=False, indent=2)
                            print(f"💾 元数据已更新")
                    except Exception as e:
                        print(f"⚠️  添加新PDF失败: {e}")
            
        except Exception as e:
            print(f"⚠️  加载向量数据库失败: {e}，将重新创建")
            vectorstore, retriever, doc_processor = None, None, None
    else:
        vectorstore, retriever, doc_processor = None, None, None
    
    # 如果没有加载成功，则创建新的向量数据库
    if vectorstore is None:
        print("🔧 正在创建新的向量数据库...")
        
        # 初始化文档处理器
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
                
                # 创建向量数据库（带持久化）
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                from config import EMBEDDING_MODEL, COLLECTION_NAME
                
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                
                vectorstore = Chroma.from_documents(
                    documents=doc_splits,
                    embedding=embeddings,
                    collection_name=COLLECTION_NAME,
                    persist_directory=persist_dir  # 持久化目录
                )
                
                retriever = vectorstore.as_retriever()
                
                print(f"✅ PDF处理完成，共 {len(doc_splits)} 个文档块")
                print(f"💾 向量数据库已持久化到: {persist_dir}")
                
                # 保存元数据
                file_hash = get_file_hash(pdf_path)
                if file_hash:
                    processed_files[file_hash] = {
                        'path': pdf_path,
                        'type': 'pdf',
                        'chunks': len(doc_splits),
                        'processed_at': time.time()
                    }
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump({'processed_files': processed_files}, f, ensure_ascii=False, indent=2)
                    print(f"💾 元数据已保存")
                
            except Exception as e:
                print(f"❌ PDF处理失败: {e}")
                return None, None
        else:
            # 使用默认知识库
            print("📄 使用默认知识库...")
            try:
                vectorstore, retriever, doc_splits = doc_processor.setup_knowledge_base()
                
                # 将默认知识库也持久化
                if vectorstore and hasattr(vectorstore, '_persist_directory'):
                    vectorstore._persist_directory = persist_dir
                    print(f"💾 默认知识库已持久化到: {persist_dir}")
                    
            except Exception as e:
                print(f"❌ 默认知识库加载失败: {e}")
                return None, None
    
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
            # 跳过隐藏文件和空文件名
            if not file or file.startswith('.'):
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
    
    # 过滤有效的PDF文件（排除隐藏文件）
    try:
        all_files = os.listdir(working_dir)
        
        # 修复：移除文件名长度限制，支持 .pdf 等短文件名
        pdf_files = [
            f for f in all_files 
            if f.lower().endswith('.pdf')  # 小写比较
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
        all_files = []
    
    print(f"\n📁 /kaggle/working/ 中的文件:")
    
    # 调试：详细显示所有文件和过滤过程
    print("\n🔍 详细调试信息:")
    print(f"   目录中总共 {len(all_files)} 个项目")
    for f in all_files:
        f_path = os.path.join(working_dir, f)
        is_file = os.path.isfile(f_path)
        is_dir = os.path.isdir(f_path)
        f_lower = f.lower()
        
        # 检查 PDF
        if f_lower.endswith('.pdf'):
            file_size = os.path.getsize(f_path) if is_file else 0
            print(f"   📄 {f}: 是文件={is_file}, 大小={file_size/1024:.1f}KB, 长度={len(f)}")
        # 检查图片
        elif any(f_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
            file_size = os.path.getsize(f_path) if is_file else 0
            print(f"   🖼️ {f}: 是文件={is_file}, 大小={file_size/1024:.1f}KB")
        else:
            print(f"   ⚪ {f}: 类型={'[目录]' if is_dir else '[文件]'}")
    
    print(f"\n📊 过滤结果:")
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