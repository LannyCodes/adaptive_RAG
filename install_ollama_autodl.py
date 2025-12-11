import requests
import os
import subprocess
import sys

def install_ollama():
    print("🔍正在尝试自动寻找 Ollama 的最新版本...")
    
    # 1. 获取最新 Release 信息
    # AutoDL 环境访问 GitHub API 可能不稳定，我们尝试使用镜像或直接访问
    # 如果直接访问失败，我们将尝试几个已知的最新版本硬编码
    
    download_url = ""
    filename = "ollama-linux-amd64.tgz" # 默认假设是 tgz
    
    try:
        # 尝试访问 GitHub API (可能需要代理，这里先试直连，不行就用备用逻辑)
        print("   正在请求 GitHub API 获取最新下载地址...")
        api_url = "https://api.github.com/repos/ollama/ollama/releases/latest"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            tag_name = data.get("tag_name")
            print(f"✅ 发现最新版本: {tag_name}")
            
            # 寻找合适的 asset
            for asset in data.get("assets", []):
                name = asset.get("name")
                if "linux-amd64.tgz" in name:
                    download_url = asset.get("browser_download_url")
                    filename = name
                    break
                elif "linux-amd64" in name and "rocm" not in name and "tgz" not in name:
                    # 备选：如果是纯二进制文件
                    download_url = asset.get("browser_download_url")
                    filename = name
            
            if download_url:
                # 添加代理前缀
                download_url = "https://mirror.ghproxy.com/" + download_url
                print(f"   构建镜像下载地址: {download_url}")
        else:
            print(f"⚠️ 无法获取最新版本信息 (HTTP {response.status_code})，将尝试使用硬编码的备用版本。")
            
    except Exception as e:
        print(f"⚠️ 访问 GitHub API 失败: {e}。将尝试使用硬编码的备用版本。")

    # 2. 如果 API 失败，使用硬编码的最新已知稳定版 (v0.5.1 / v0.4.x)
    #    注意：Ollama 版本迭代快，这里我们尝试构建一个大概率存在的地址
    if not download_url:
        print("   使用备用下载策略...")
        # 尝试构建 latest 的直接下载链接 (通过 ghproxy)
        # 通常 asset 名字是 ollama-linux-amd64.tgz 或 ollama-linux-amd64
        # 我们先试 tgz
        download_url = "https://mirror.ghproxy.com/https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz"

    # 3. 执行下载
    print(f"⬇️ 开始下载: {filename} ...")
    try:
        # 使用 curl 下载，因为它在 shell 中显示进度条更直观，或者我们用 python stream
        # 这里用 subprocess 调用 wget，因为它更健壮
        subprocess.run(["wget", "-O", filename, download_url], check=True)
        print("✅ 下载完成。")
        
        # 4. 安装
        print("📦 正在安装...")
        
        # 判断是 tgz 还是二进制
        if filename.endswith(".tgz"):
            subprocess.run(["sudo", "tar", "-C", "/usr", "-xzf", filename], check=True)
        else:
            # 假设是二进制
            subprocess.run(["chmod", "+x", filename], check=True)
            subprocess.run(["sudo", "mv", filename, "/usr/bin/ollama"], check=True)
            
        print("🎉 Ollama 安装成功！")
        
        # 5. 验证
        subprocess.run(["ollama", "--version"])
        
        # 6. 启动提示
        print("\n🚀 请运行以下命令启动服务：")
        print("nohup ollama serve > ollama.log 2>&1 &")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载或安装过程中出错: {e}")
        # 如果是 404，提示用户尝试另一个文件名
        if "404" in str(e) or os.path.getsize(filename) < 1000:
             print("⚠️ 可能是文件名不对，正在尝试下载纯二进制版本...")
             try:
                 alt_url = "https://mirror.ghproxy.com/https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64"
                 subprocess.run(["wget", "-O", "ollama-linux-amd64", alt_url], check=True)
                 subprocess.run(["chmod", "+x", "ollama-linux-amd64"], check=True)
                 subprocess.run(["sudo", "mv", "ollama-linux-amd64", "/usr/bin/ollama"], check=True)
                 print("🎉 Ollama (二进制版) 安装成功！")
             except Exception as e2:
                 print(f"❌ 还是失败了: {e2}")

if __name__ == "__main__":
    install_ollama()
