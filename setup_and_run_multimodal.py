#!/usr/bin/env python3
"""
环境配置和运行脚本 (自适应RAG多模态版)
配置环境并运行 main.py (包含多模态检索功能)

使用方法:
  终端:   python setup_and_run_multimodal.py
  Kaggle: 直接在 Notebook 单元格中运行即可（自动检测环境）
"""

import os
import sys

print("="*60)
print("🚀 自适应RAG (多模态) 环境配置和运行")
print("="*60)

# ============================================================
# 1. 配置环境
# ============================================================
def setup_environment():
    """配置环境变量"""
    print("\n⚙️ 步骤 1/2: 配置环境变量...")
    
    # 检查.env文件
    if os.path.exists(".env"):
        print("   ✅ 发现 .env 文件，加载配置...")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("   ✅ 环境变量已加载")
        except ImportError:
            print("   ⚠️ python-dotenv 未安装，跳过 .env 加载")
    else:
        print("   ℹ️ 未找到 .env 文件")
    
    # 显示环境变量状态
    print("\n   📋 环境变量状态:")
    print(f"      • TAVILY_API_KEY: {'✅ 已设置' if os.environ.get('TAVILY_API_KEY') else '⚠️ 未设置'}")
    print(f"      • NOMIC_API_KEY: {'✅ 已设置' if os.environ.get('NOMIC_API_KEY') else '⚠️ 未设置'}")
    
    # 添加当前目录到 Python 路径
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"\n   ✅ 已添加到 Python 路径: {current_dir}")
    
    # 检测运行环境
    is_notebook = _is_notebook_env()
    if is_notebook:
        print("\n   📓 检测到 Notebook/Kaggle 环境 → 将使用 ipywidgets 交互模式")
    else:
        print("\n   💻 检测到终端环境 → 将使用命令行交互模式")
    
    print("\n   💡 注意: 多模态功能需要Pillow库，请确保已安装")

# ============================================================
# 环境检测
# ============================================================
def _is_notebook_env():
    """检测是否运行在 Jupyter/Kaggle Notebook 环境中"""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell_class = type(shell).__name__
            if 'ZMQ' in shell_class or 'Colab' in shell_class:
                return True
    except (ImportError, NameError):
        pass

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.path.exists('/kaggle'):
        return True

    try:
        if not sys.stdin.isatty() and 'ipykernel' in sys.modules:
            return True
    except Exception:
        pass

    return False

# ============================================================
# 2. 运行 main.py（进程内导入，兼容 Notebook 交互）
# ============================================================
def run_main_multimodal():
    """在当前进程中导入并运行 main.py（而非子进程）"""
    print("\n🚀 步骤 2/2: 运行 main.py (多模态自适应检索)...")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists("main.py"):
        print("\n❌ 错误: 未找到 main.py 文件")
        print("   请确保在正确的目录中运行此脚本")
        return False
    
    print("\n🔄 启动自适应RAG系统...\n")
    
    try:
        # 在当前进程中导入 main 模块并直接调用 main()
        # 这样 ipywidgets 才能渲染到 Notebook 中
        import main as main_module
        main_module.main()
        return True
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
        return False
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# 主函数
# ============================================================
def main():
    """主执行流程"""
    try:
        # 1. 配置环境
        setup_environment()
        
        # 2. 运行 main.py
        run_main_multimodal()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
