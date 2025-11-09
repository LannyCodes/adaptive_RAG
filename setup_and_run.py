#!/usr/bin/env python3
"""
环境配置和运行脚本
简化版：只负责配置环境和运行 main_graphrag.py

使用方法:
python colab_setup_and_run.py
"""

import os
import sys
import subprocess

print("="*60)
print("🚀 GraphRAG 环境配置和运行")
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
    
    print("\n   💡 注意: 新增的多模态功能需要Pillow库，请确保已安装")

# ============================================================
# 2. 运行 main_graphrag.py
# ============================================================
def run_main_graphrag():
    """运行 main.py"""
    # print("\n🚀 步骤 2/2: 运行 main_graphrag.py...")
    print("\n🚀 步骤 2/2: 运行 main.py...")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists("main.py"):
        print("\n❌ 错误: 未找到 main.py 文件")
        print("   请确保在正确的目录中运行此脚本")
        return False
    
    print("\n🔄 启动 GraphRAG...\n")
    
    try:
        # 运行 main.py
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=False,  # 实时显示输出
        )
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✅ 运行成功！")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print(f"❌ 运行失败 (返回码: {result.returncode})")
            print("="*60)
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
        return False
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        return False

# ============================================================
# 主函数
# ============================================================
def main():
    """主执行流程"""
    try:
        # 1. 配置环境
        setup_environment()
        
        # 2. 运行 main_graphrag.py
        success = run_main_graphrag()
        
        if success:
            print("\n💡 提示: 生成的知识图谱保存在配置的路径中")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
