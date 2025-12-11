# AutoDL 极速安装 Ollama 脚本 (国内源版)
# 使用清华/阿里/腾讯等国内镜像源或者国内云存储的二进制文件
# 彻底避开 GitHub 和 ghproxy 的 SSL 问题

import subprocess
import os
import sys

def install_from_china_mirror():
    print("🇨🇳 正在尝试使用国内极速源安装 Ollama...")
    
    # 这是一个托管在国内 CDN 的 Ollama 二进制文件 (v0.1.32 版本)
    # 如果这个链接失效，可以尝试 ModelScope 或者其他国内 AI 社区的源
    # 这里我们使用一个比较稳定的第三方国内源，或者尝试使用 pip 安装 ollama 库虽然它是客户端
    
    # 方案 A: 尝试使用 ModelScope 的下载链接 (如果存在) 或其他国内大厂源
    # 目前最稳妥的是直接下载二进制，这里提供几个备选
    
    urls = [
        # 备选源 1: 某国内 AI 社区镜像 (速度快)
        "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/ollama-linux-amd64.tgz",
        # 备选源 2: 使用 http 协议绕过 SSL (ghproxy 的 http 端口)
        "http://mirror.ghproxy.com/https://github.com/ollama/ollama/releases/download/v0.1.32/ollama-linux-amd64.tgz"
    ]
    
    filename = "ollama-linux-amd64.tgz"
    
    for url in urls:
        print(f"⬇️ 尝试下载: {url}")
        try:
            # 使用 wget，添加 --no-check-certificate 忽略 SSL 错误
            # 添加 -c 断点续传
            subprocess.run(["wget", "--no-check-certificate", "-c", "-O", filename, url], check=True)
            
            print("📦 下载成功，开始解压...")
            subprocess.run(["sudo", "tar", "-C", "/usr", "-xzf", filename], check=True)
            
            print("🎉 Ollama 安装成功！")
            subprocess.run(["ollama", "--version"])
            print("\n🚀 启动命令: nohup ollama serve > ollama.log 2>&1 &")
            return
            
        except subprocess.CalledProcessError:
            print(f"❌ 从 {url} 下载失败，尝试下一个源...")
            if os.path.exists(filename):
                os.remove(filename) # 清理失败的文件

    print("⚠️ 所有源都失败了。")
    print("建议手动上传：请在本地下载好 ollama-linux-amd64.tgz，然后通过 AutoDL 网页的文件上传功能传上去。")

if __name__ == "__main__":
    # 先清理一下旧的
    if os.path.exists("ollama-linux-amd64.tgz"):
        os.remove("ollama-linux-amd64.tgz")
    install_from_china_mirror()
