"""
Kaggle Gradio 智能问答界面
适合在 Kaggle Notebook 中运行的交互式 RAG 系统
"""

import gradio as gr
import sys
import os

# 添加项目路径
if '/kaggle/working/adaptive_RAG' not in sys.path:
    sys.path.insert(0, '/kaggle/working/adaptive_RAG')

from main import AdaptiveRAGSystem


class RAGChatInterface:
    """RAG聊天界面"""
    
    def __init__(self):
        """初始化RAG系统"""
        print("🚀 正在初始化RAG系统...")
        try:
            self.rag_system = AdaptiveRAGSystem()
            self.initialized = True
            print("✅ RAG系统初始化成功")
        except Exception as e:
            print(f"❌ RAG系统初始化失败: {e}")
            self.initialized = False
            self.error_message = str(e)
    
    def chat(self, message, history):
        """
        处理聊天消息
        
        Args:
            message: 用户输入的消息
            history: 聊天历史记录
            
        Returns:
            响应消息
        """
        if not self.initialized:
            return f"❌ 系统未初始化: {self.error_message}"
        
        if not message or not message.strip():
            return "⚠️ 请输入有效的问题"
        
        try:
            # 查询RAG系统
            result = self.rag_system.query(message, verbose=False)
            
            # 构建响应
            answer = result.get('answer', '无法生成答案')
            
            # 添加评估指标（可选）
            metrics = result.get('retrieval_metrics')
            if metrics:
                metrics_text = f"\n\n📊 检索指标:\n"
                metrics_text += f"- 耗时: {metrics.get('latency', 0):.2f}秒\n"
                metrics_text += f"- 文档数: {metrics.get('retrieved_docs_count', 0)}\n"
                metrics_text += f"- Precision@3: {metrics.get('precision_at_3', 0):.2f}\n"
                # answer += metrics_text  # 取消注释以显示指标
            
            return answer
            
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 自定义CSS样式
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .chatbot {
            height: 500px;
        }
        """
        
        # 创建聊天界面
        with gr.Blocks(css=custom_css, title="🤖 自适应RAG智能问答") as demo:
            gr.Markdown(
                """
                # 🤖 自适应RAG智能问答系统
                
                基于LangGraph的自适应检索增强生成系统，支持：
                - 🔍 智能路由（本地知识库 vs 网络搜索）
                - 📚 混合检索（向量 + BM25）
                - 🎯 多重质量控制（文档评分、幻觉检测）
                - 🔄 自适应查询重写
                
                **使用方法**: 在下方输入框输入问题，系统会自动选择最佳检索策略并生成答案。
                """
            )
            
            # 聊天界面
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500,
                show_label=True,
                avatar_images=(None, "🤖")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入问题",
                    placeholder="例如: AlphaCodium论文讲的是什么？",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("🚀 发送", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", scale=1)
                
            # 示例问题
            gr.Examples(
                examples=[
                    "AlphaCodium论文讲的是什么？",
                    "解释embedding嵌入的原理",
                    "什么是LLM Agent？",
                    "如何防止LLM产生幻觉？",
                    "Prompt Engineering的最佳实践是什么？"
                ],
                inputs=msg,
                label="💡 示例问题"
            )
            
            # 状态信息
            if self.initialized:
                gr.Markdown("✅ **系统状态**: 运行正常")
            else:
                gr.Markdown(f"❌ **系统状态**: 初始化失败 - {self.error_message}")
            
            # 事件绑定
            def respond(message, chat_history):
                """响应用户消息"""
                if not message:
                    return "", chat_history
                
                # 添加用户消息到历史
                chat_history.append((message, None))
                
                # 获取AI响应
                bot_message = self.chat(message, chat_history)
                
                # 更新历史
                chat_history[-1] = (message, bot_message)
                
                return "", chat_history
            
            # 绑定发送按钮
            submit_btn.click(
                respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            # 绑定回车键
            msg.submit(
                respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            # 清空对话
            clear_btn.click(lambda: None, None, chatbot, queue=False)
        
        return demo


def launch_app(share=False, server_port=7860):
    """
    启动Gradio应用
    
    Args:
        share: 是否创建公开链接（Kaggle中建议False）
        server_port: 服务器端口
    """
    print("=" * 60)
    print("🚀 启动 Gradio RAG 智能问答系统")
    print("=" * 60)
    
    # 检查是否在Kaggle环境
    is_kaggle = os.path.exists('/kaggle/working')
    
    if is_kaggle:
        print("🎯 检测到 Kaggle 环境")
        print("💡 提示: 运行后会显示本地URL")
    
    # 创建界面
    interface = RAGChatInterface()
    demo = interface.create_interface()
    
    # 启动服务
    print(f"\n🌐 正在启动服务...")
    
    # Kaggle 环境特殊配置
    if is_kaggle:
        # 在 Kaggle 中必须使用 share=True 或 inline 模式
        demo.launch(
            share=True,  # Kaggle 中强制使用 share
            server_name="0.0.0.0",
            server_port=server_port,
            show_error=True,
            inline=True,  # 内嵌显示
            quiet=False
        )
    else:
        # 本地环境配置
        demo.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0",
            show_error=True,
            quiet=False
        )


if __name__ == "__main__":
    # 在Kaggle Notebook中运行时自动启动
    launch_app(share=False)
