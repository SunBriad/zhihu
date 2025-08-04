import os
import streamlit.web.cli as stcli
import sys

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 运行Streamlit应用
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())
import sys
import subprocess
import logging
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        "streamlit", 
        "langchain", 
        "langchain-openai", 
        "faiss-cpu", 
        "playwright", 
        "python-dotenv",
        "langgraph"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """安装缺失的依赖"""
    logger.info(f"正在安装缺失的依赖: {', '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    
    # 安装 Playwright 浏览器
    if "playwright" in packages:
        logger.info("正在安装 Playwright 浏览器...")
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])

def check_env_variables():
    """检查必要的环境变量"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    return missing_vars

def create_env_file():
    """创建.env文件模板"""
    if not os.path.exists(".env"):
        with open(".env", "w", encoding="utf-8") as f:
            f.write("""# OpenAI API密钥
OPENAI_API_KEY=sk-your-openai-key

# 知乎Cookie路径
ZHIHU_COOKIE_PATH=cookies/zhihu_cookies.json
""")
        logger.info("已创建.env文件模板，请编辑填写必要的环境变量")

def ensure_directories():
    """确保必要的目录存在"""
    directories = ["backend/temp", "backend/vector_store", "cookies"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"已创建目录: {directory}")

def main():
    """主函数"""
    print("=" * 50)
    print("知乎热榜AI助手启动程序")
    print("=" * 50)
    
    # 确保必要的目录存在
    ensure_directories()
    
    # 创建.env文件模板（如果不存在）
    create_env_file()
    
    # 检查依赖
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"检测到缺失的依赖: {', '.join(missing_packages)}")
        choice = input("是否自动安装这些依赖? (y/n): ")
        if choice.lower() == 'y':
            install_dependencies(missing_packages)
        else:
            print(f"请手动安装依赖: pip install {' '.join(missing_packages)}")
            return
    
    # 检查环境变量
    missing_vars = check_env_variables()
    if missing_vars:
        print(f"缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请在.env文件中设置这些变量")
        return
    
    # 启动Streamlit应用
    print("正在启动知乎热榜AI助手...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])

if __name__ == "__main__":
    main()