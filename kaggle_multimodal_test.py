"""
Kaggle多模态测试脚本
用于在Kaggle环境中上传PDF和图片并测试多模态功能
"""

import os
import sys
import subprocess
import time
import ipywidgets as widgets
from IPython.display import display, HTML
from io import BytesIO
import base64
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, '/kaggle/working/adaptive_RAG')

# 导入项目模块
from document_processor import DocumentProcessor
from main import AdaptiveRAGSystem
from config import ENABLE_MULTIMODAL, SUPPORTED_IMAGE_FORMATS

class KaggleMultimodalUploader:
    """Kaggle多模态文件上传和处理类"""
    
    def __init__(self):
        """初始化上传器"""
        self.uploaded_files = {}
        self.doc_processor = None
        self.rag_system = None
        self.setup_system()
        
    def setup_system(self):
        """设置RAG系统"""
        print("🔧 正在初始化自适应RAG系统...")
        
        # 初始化文档处理器
        self.doc_processor = DocumentProcessor()
        
        # 初始化RAG系统
        self.rag_system = AdaptiveRAGSystem()
        
        print("✅ 系统初始化完成")
    
    def create_upload_widgets(self):
        """创建文件上传小部件"""
        # PDF上传小部件
        pdf_upload = widgets.FileUpload(
            accept='.pdf',
            multiple=False,
            description='上传PDF',
            style={'description_width': 'initial'}
        )
        
        # 图片上传小部件
        image_upload = widgets.FileUpload(
            accept='.jpg,.jpeg,.png,.gif,.bmp',
            multiple=True,
            description='上传图片',
            style={'description_width': 'initial'}
        )
        
        # 处理按钮
        process_button = widgets.Button(
            description='处理文件',
            button_style='success',
            tooltip='点击处理上传的文件'
        )
        
        # 查询输入框
        query_input = widgets.Text(
            value='',
            placeholder='输入您的问题...',
            description='问题:',
            style={'description_width': 'initial'}
        )
        
        # 查询按钮
        query_button = widgets.Button(
            description='查询',
            button_style='info',
            tooltip='点击提交查询'
        )
        
        # 输出区域
        output_area = widgets.Output()
        
        # 绑定事件处理函数
        pdf_upload.observe(self.on_pdf_upload, names='value')
        image_upload.observe(self.on_image_upload, names='value')
        process_button.on_click(self.on_process_click)
        query_button.on_click(self.on_query_click)
        
        # 显示小部件
        display(HTML("<h2>📄 PDF上传</h2>"))
        display(pdf_upload)
        
        display(HTML("<h2>🖼️ 图片上传</h2>"))
        display(image_upload)
        
        display(HTML("<h2>🔧 文件处理</h2>"))
        display(process_button)
        
        display(HTML("<h2>❓ 查询</h2>"))
        display(query_input)
        display(query_button)
        
        display(HTML("<h2>📋 输出</h2>"))
        display(output_area)
        
        # 保存小部件引用
        self.pdf_upload = pdf_upload
        self.image_upload = image_upload
        self.process_button = process_button
        self.query_input = query_input
        self.query_button = query_button
        self.output_area = output_area
    
    def on_pdf_upload(self, change):
        """处理PDF上传事件"""
        uploaded_file = list(change['new'].values())[0]
        filename = uploaded_file['name']
        content = uploaded_file['content']
        
        # 保存文件
        pdf_path = f'/kaggle/working/{filename}'
        with open(pdf_path, 'wb') as f:
            f.write(content)
        
        self.uploaded_files['pdf'] = pdf_path
        print(f"✅ PDF已上传: {filename}")
    
    def on_image_upload(self, change):
        """处理图片上传事件"""
        uploaded_files = change['new']
        image_paths = []
        
        for filename, file_info in uploaded_files.items():
            # 保存文件
            img_path = f'/kaggle/working/{filename}'
            with open(img_path, 'wb') as f:
                f.write(file_info['content'])
            image_paths.append(img_path)
        
        self.uploaded_files['images'] = image_paths
        print(f"✅ 已上传 {len(image_paths)} 张图片")
    
    def on_process_click(self, b):
        """处理文件按钮点击事件"""
        with self.output_area:
            self.output_area.clear_output()
            
            if 'pdf' not in self.uploaded_files:
                print("⚠️ 请先上传PDF文件")
                return
            
            print("🔧 正在处理PDF文件...")
            pdf_path = self.uploaded_files['pdf']
            
            try:
                # 加载PDF文档
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # 分割文档
                doc_splits = self.doc_processor.split_documents(docs)
                
                # 创建向量数据库
                vectorstore, retriever = self.doc_processor.create_vectorstore(doc_splits)
                
                # 更新RAG系统的检索器
                self.rag_system.retriever = retriever
                self.rag_system.doc_processor = self.doc_processor
                self.rag_system.workflow_nodes.retriever = retriever
                self.rag_system.workflow_nodes.doc_processor = self.doc_processor
                
                print(f"✅ PDF处理完成，共 {len(doc_splits)} 个文档块")
                
            except Exception as e:
                print(f"❌ PDF处理失败: {e}")
    
    def on_query_click(self, b):
        """查询按钮点击事件"""
        with self.output_area:
            self.output_area.clear_output()
            
            query = self.query_input.value
            if not query:
                print("⚠️ 请输入查询内容")
                return
            
            print(f"🔍 查询: {query}")
            
            try:
                # 获取图片路径（如果有）
                image_paths = self.uploaded_files.get('images', [])
                
                # 执行多模态查询
                if ENABLE_MULTIMODAL and image_paths:
                    print(f"🖼️ 使用 {len(image_paths)} 张图片进行多模态查询")
                    result = self.rag_system.query(query)
                else:
                    print("📄 使用文本查询")
                    result = self.rag_system.query(query)
                
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
                
            except Exception as e:
                print(f"❌ 查询失败: {e}")


def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("🔧 设置Kaggle环境...")
    
    # 安装必要的依赖
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                   'ipywidgets', 'PyPDF2', 'pdfplumber', 'Pillow'])
    
    # 启用ipywidgets
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except:
        pass
    
    print("✅ 环境设置完成")


def main():
    """主函数"""
    # 设置环境
    setup_kaggle_environment()
    
    # 创建上传器实例
    uploader = KaggleMultimodalUploader()
    
    # 创建并显示上传小部件
    uploader.create_upload_widgets()
    
    print("\n🎉 多模态测试界面已准备就绪!")
    print("💡 使用说明:")
    print("   1. 上传PDF文件")
    print("   2. (可选) 上传相关图片")
    print("   3. 点击'处理文件'按钮")
    print("   4. 输入问题并点击'查询'")


if __name__ == "__main__":
    main()