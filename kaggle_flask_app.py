"""
Kaggle Flask 智能问答界面
在 Kaggle Notebook 中使用 Flask 创建交互式 RAG 系统
"""

from flask import Flask, render_template_string, request, jsonify
import sys
import os
import threading

# 添加项目路径
if '/kaggle/working/adaptive_RAG' not in sys.path:
    sys.path.insert(0, '/kaggle/working/adaptive_RAG')

from main import AdaptiveRAGSystem

# 创建Flask应用
app = Flask(__name__)

# 全局RAG系统实例
rag_system = None
initialization_error = None


# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 自适应RAG智能问答</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 800px;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f7f7f7;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 18px;
            position: relative;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: #667eea;
            color: white;
            margin-left: auto;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
        }
        
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            margin: 0 10px;
        }
        
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        #question-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #question-input:focus {
            border-color: #667eea;
        }
        
        #send-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: transform 0.2s;
        }
        
        #send-btn:hover {
            transform: scale(1.05);
        }
        
        #send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .examples {
            padding: 20px;
            background: #f0f0f0;
        }
        
        .examples h3 {
            margin-bottom: 10px;
            color: #667eea;
        }
        
        .example-btn {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background: white;
            border: 1px solid #667eea;
            color: #667eea;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        
        .example-btn:hover {
            background: #667eea;
            color: white;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
            color: #667eea;
        }
        
        .error-message {
            background: #fee;
            color: #c33;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 4px solid #c33;
        }
        
        .status {
            padding: 10px 20px;
            text-align: center;
            font-size: 0.9em;
        }
        
        .status.ok {
            background: #d4edda;
            color: #155724;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 自适应RAG智能问答</h1>
            <p>基于 LangGraph 的检索增强生成系统</p>
        </div>
        
        <div class="status {{ 'ok' if system_ok else 'error' }}">
            {{ '✅ 系统运行正常' if system_ok else '❌ 系统初始化失败: ' + error }}
        </div>
        
        <div class="chat-container" id="chat-container">
            <div class="message bot">
                <div class="avatar">🤖</div>
                <div class="message-content">
                    你好！我是自适应RAG智能助手。我可以帮你回答关于LLM、Agent、Prompt Engineering等问题。请输入你的问题！
                </div>
            </div>
        </div>
        
        <div class="examples">
            <h3>💡 示例问题</h3>
            <button class="example-btn" onclick="askExample('AlphaCodium论文讲的是什么？')">
                AlphaCodium论文讲的是什么？
            </button>
            <button class="example-btn" onclick="askExample('解释embedding嵌入的原理')">
                解释embedding嵌入的原理
            </button>
            <button class="example-btn" onclick="askExample('什么是LLM Agent？')">
                什么是LLM Agent？
            </button>
            <button class="example-btn" onclick="askExample('如何防止LLM产生幻觉？')">
                如何防止LLM产生幻觉？
            </button>
        </div>
        
        <div class="loading" id="loading">
            ⏳ AI正在思考中...
        </div>
        
        <div class="input-container">
            <div class="input-group">
                <input 
                    type="text" 
                    id="question-input" 
                    placeholder="输入你的问题..."
                    onkeypress="handleKeyPress(event)"
                >
                <button id="send-btn" onclick="sendQuestion()">🚀 发送</button>
            </div>
        </div>
    </div>
    
    <script>
        function addMessage(content, isUser) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
            
            messageDiv.innerHTML = `
                <div class="avatar">${isUser ? '👤' : '🤖'}</div>
                <div class="message-content">${content}</div>
            `;
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function askExample(question) {
            document.getElementById('question-input').value = question;
            sendQuestion();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendQuestion();
            }
        }
        
        async function sendQuestion() {
            const input = document.getElementById('question-input');
            const question = input.value.trim();
            
            if (!question) {
                return;
            }
            
            // 添加用户消息
            addMessage(question, true);
            input.value = '';
            
            // 显示加载状态
            const loading = document.getElementById('loading');
            const sendBtn = document.getElementById('send-btn');
            loading.style.display = 'block';
            sendBtn.disabled = true;
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addMessage(data.answer, false);
                } else {
                    addMessage('❌ 错误: ' + data.error, false);
                }
            } catch (error) {
                addMessage('❌ 网络错误: ' + error.message, false);
            } finally {
                loading.style.display = 'none';
                sendBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页"""
    return render_template_string(
        HTML_TEMPLATE,
        system_ok=(rag_system is not None),
        error=initialization_error or ""
    )


@app.route('/api/query', methods=['POST'])
def query():
    """处理查询请求"""
    if rag_system is None:
        return jsonify({
            'success': False,
            'error': f'系统未初始化: {initialization_error}'
        })
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': '问题不能为空'
            })
        
        # 查询RAG系统
        result = rag_system.query(question, verbose=False)
        answer = result.get('answer', '无法生成答案')
        
        return jsonify({
            'success': True,
            'answer': answer,
            'metrics': result.get('retrieval_metrics')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def initialize_rag():
    """初始化RAG系统"""
    global rag_system, initialization_error
    
    print("🚀 正在初始化RAG系统...")
    try:
        rag_system = AdaptiveRAGSystem()
        print("✅ RAG系统初始化成功")
    except Exception as e:
        initialization_error = str(e)
        print(f"❌ RAG系统初始化失败: {e}")


def run_flask(host='0.0.0.0', port=5000):
    """
    启动Flask应用
    
    Args:
        host: 主机地址
        port: 端口号
    """
    print("=" * 60)
    print("🚀 启动 Flask RAG 智能问答系统")
    print("=" * 60)
    
    # 初始化RAG系统
    initialize_rag()
    
    # 检查是否在Kaggle环境
    is_kaggle = os.path.exists('/kaggle/working')
    
    if is_kaggle:
        print("🎯 检测到 Kaggle 环境")
        print(f"💡 访问地址: http://localhost:{port}")
        print("⚠️  注意: Kaggle无法从外部访问，只能在Notebook内查看")
    
    print(f"\n🌐 正在启动服务...")
    print(f"   地址: http://{host}:{port}")
    print("=" * 60)
    
    # 启动Flask
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_flask(port=5000)
