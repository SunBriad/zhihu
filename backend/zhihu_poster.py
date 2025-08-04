import time
import json
import os
import logging
import pyperclip  # 用于复制粘贴操作
import webbrowser
import subprocess
import sys
import platform

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置文件路径
COOKIE_PATH = os.environ.get("ZHIHU_COOKIE_PATH", "cookies/zhihu_cookies.json")

def ensure_dir_exists(file_path):
    """确保目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")

async def save_cookies(page, path=COOKIE_PATH):
    """保存cookies到文件"""
    ensure_dir_exists(path)
    cookies = await page.context.cookies()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cookies, f, ensure_ascii=False, indent=2)
    logger.info(f"Cookies已保存到: {path}")

async def load_cookies(page, path=COOKIE_PATH):
    """从文件加载cookies"""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            await page.context.add_cookies(cookies)
            logger.info(f"已从 {path} 加载cookies")
            return True
        else:
            logger.warning(f"Cookie文件不存在: {path}")
            return False
    except Exception as e:
        logger.error(f"加载cookies时出错: {str(e)}")
        return False

async def login_zhihu(page):
    """登录知乎"""
    await page.goto("https://www.zhihu.com/signin")
    logger.info("已打开知乎登录页面")
    
    # 等待用户手动登录
    logger.info("请在浏览器中手动完成登录...")
    
    # 等待登录完成
    await page.wait_for_selector(".AppHeader-profile", timeout=300000)  # 5分钟超时
    logger.info("登录成功")
    
    # 保存cookies
    await save_cookies(page)

def open_browser_with_question(question: str, answer: str):
    """打开浏览器并导航到知乎搜索页面"""
    try:
        # 保存回答到剪贴板
        pyperclip.copy(answer)
        logger.info("已将回答内容复制到剪贴板")
        
        # 构建搜索URL
        search_url = f"https://www.zhihu.com/search?type=content&q={question}"
        
        # 打开浏览器
        logger.info(f"正在打开浏览器，搜索问题: {question}")
        webbrowser.open(search_url)
        
        # 显示指导信息
        print("\n=== 知乎回答发布指南 ===")
        print("1. 在打开的浏览器中，找到并点击相关问题")
        print("2. 点击\"写回答\"按钮")
        print("3. 在编辑器中按 Ctrl+V 粘贴回答内容（已自动复制到剪贴板）")
        print("4. 点击\"发布回答\"按钮完成发布")
        print("========================\n")
        
        return True
    except Exception as e:
        logger.error(f"打开浏览器时出错: {str(e)}")
        return False

def post_to_zhihu(question: str, answer: str):
    """发布回答到知乎"""
    try:
        # 保存回答到文件
        save_path = f"zhihu_answers/{question[:20]}.md"
        ensure_dir_exists(save_path)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"# {question}\n\n{answer}")
        logger.info(f"回答已保存到文件: {save_path}")
        
        # 打开浏览器引导用户手动发布
        browser_opened = open_browser_with_question(question, answer)
        
        # 提示用户
        print(f"\n回答已保存到文件: {save_path}")
        if not browser_opened:
            print("自动打开浏览器失败，请手动访问知乎并发布回答。")
        
        return browser_opened
    except Exception as e:
        logger.error(f"发布回答时出错: {str(e)}")
        return False

def test_zhihu_login():
    """测试知乎登录功能"""
    try:
        # 直接打开知乎首页
        webbrowser.open("https://www.zhihu.com")
        logger.info("已打开知乎首页，请检查登录状态")
        return True
    except Exception as e:
        logger.error(f"测试登录时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 测试登录功能
    test_zhihu_login()