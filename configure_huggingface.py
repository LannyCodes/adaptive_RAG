#!/usr/bin/env python3
"""
Hugging Face Vectara 模型访问配置
解决 Vectara 幻觉检测模型的权限问题

使用方法:
1. python configure_huggingface.py
2. 按照提示进行 Hugging Face 登录和模型访问权限申请
"""

import os
import subprocess
import sys

def check_huggingface_login():
    """检查是否已登录 Hugging Face"""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def login_huggingface():
    """登录 Hugging Face"""
    print("🔐 需要登录 Hugging Face 才能访问 Vectara 模型")
    print("\n📋 登录步骤:")
    print("1. 访问 https://huggingface.co/join 注册账户（如果还没有）")
    print("2. 访问 https://huggingface.co/settings/tokens 获取访问令牌")
    print("3. 运行以下命令登录:")
    print("   huggingface-cli login")
    
    choice = input("\n是否现在尝试登录? (y/n): ").lower().strip()
    if choice == 'y':
        try:
            subprocess.run(["huggingface-cli", "login"], check=True)
            print("✅ Hugging Face 登录成功!")
            return True
        except subprocess.CalledProcessError:
            print("❌ 登录失败，请手动执行: huggingface-cli login")
            return False
        except FileNotFoundError:
            print("❌ 未找到 huggingface-cli，请安装: pip install huggingface_hub")
            return False
    return False

def check_model_access():
    """检查是否可以访问 Vectara 模型"""
    print("\n🔍 检查 Vectara 模型访问权限...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("vectara/hallucination_evaluation_model")
        print("✅ 可以访问 Vectara 模型!")
        return True
    except Exception as e:
        print(f"❌ 无法访问 Vectara 模型: {e}")
        print("\n📋 需要完成以下步骤:")
        print("1. 访问: https://huggingface.co/vectara/hallucination_evaluation_model")
        print("2. 点击页面上的 'Agree and access repository' 按钮")
        print("3. 确保已登录 Hugging Face 账户")
        return False

def update_config_disable_vectara():
    """更新配置，禁用 Vectara 模型，使用 NLI 方法"""
    print("\n⚙️ 更新配置，禁用 Vectara 模型...")
    
    config_file = "hallucination_config.py"
    if os.path.exists(config_file):
        # 读取原始配置
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份原始配置
        with open(f"{config_file}.backup", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 修改配置，只使用 NLI 方法
        updated_content = content.replace(
            'HALLUCINATION_DETECTION_METHOD = "hybrid"',
            'HALLUCINATION_DETECTION_METHOD = "nli"'
        )
        
        # 保存更新后的配置
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("✅ 配置已更新，将使用 NLI 幻觉检测方法")
        return True
    else:
        print(f"❌ 未找到配置文件: {config_file}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("🔧 Hugging Face Vectara 模型访问配置")
    print("="*60)
    
    # 1. 检查是否已登录
    if not check_huggingface_login():
        print("\n⚠️ 未检测到 Hugging Face 登录状态")
        if not login_huggingface():
            print("\n❌ 无法完成登录，将使用备选方案")
            update_config_disable_vectara()
            return
    
    # 2. 检查模型访问权限
    if not check_model_access():
        print("\n❌ 无法访问 Vectara 模型")
        
        choice = input("\n选择解决方案:\n1. 手动申请模型访问权限后重试\n2. 禁用 Vectara，使用 NLI 方法\n\n请输入选择 (1/2): ").strip()
        
        if choice == "1":
            print("\n📋 手动申请步骤:")
            print("1. 访问: https://huggingface.co/vectara/hallucination_evaluation_model")
            print("2. 登录您的 Hugging Face 账户")
            print("3. 点击 'Agree and access repository' 按钮")
            print("4. 重新运行此脚本验证")
        elif choice == "2":
            update_config_disable_vectara()
        else:
            print("❌ 无效选择")
    else:
        print("\n✅ 所有检查通过，可以使用 Vectara 模型!")

if __name__ == "__main__":
    main()