import requests
import logging
import random
import json
import time
import re
from bs4 import BeautifulSoup

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_zhihu_hot_questions(limit=10):
    """
    获取知乎热榜问题
    
    Args:
        limit: 返回的问题数量
        
    Returns:
        list: 热榜问题列表
    """
    # 尝试多种方法获取知乎热榜
    
    methods = [
        get_zhihu_hot_via_api,
        get_zhihu_hot_via_web,
        get_zhihu_hot_via_search
    ]
    
    for method in methods:
        try:
            questions = method(limit)
            if questions and len(questions) > 0:
                return questions
        except Exception as e:
            logger.warning(f"方法 {method.__name__} 获取知乎热榜失败: {str(e)}")
    
    # 如果所有方法都失败，返回备用问题
    logger.warning("所有获取知乎热榜的方法都失败，使用备用问题")
    return get_fallback_questions(limit)

def get_zhihu_hot_via_api(limit=10):
    """通过知乎API获取热榜问题"""
    try:
        # 知乎热榜API
        url = "https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit=50"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Origin': 'https://www.zhihu.com',
            'Referer': 'https://www.zhihu.com/hot',
        }
        
        logger.info(f"正在通过API请求知乎热榜: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"API请求知乎热榜失败，状态码: {response.status_code}")
            return []
        
        data = response.json()
        questions = []
        
        if 'data' in data:
            for item in data['data']:
                if 'target' in item and 'title' in item['target']:
                    title = item['target']['title']
                    # 只保留问号结尾的内容（问题）
                    if "？" in title or "?" in title:
                        questions.append(title)
                
                # 如果已经收集足够的问题，就停止
                if len(questions) >= limit:
                    break
        
        logger.info(f"通过API成功获取 {len(questions)} 个知乎热榜问题")
        
        # 如果没有获取到足够的问题，使用备用问题
        if len(questions) < limit:
            logger.warning(f"从API只获取到 {len(questions)} 个问题，将添加备用问题")
            questions.extend(get_fallback_questions(limit - len(questions)))
        
        return questions[:limit]
    except Exception as e:
        logger.error(f"通过API获取知乎热榜问题时出错: {str(e)}")
        return []

def get_zhihu_hot_via_web(limit=10):
    """通过网页获取知乎热榜问题"""
    try:
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Cookie': '_zap=a7389b92-41f3-4eb2-9595-b11d7f5a0a2d'  # 添加一个基本的Cookie
        }
        
        # 知乎热榜页面URL
        url = "https://www.zhihu.com/hot"
        
        # 发送请求
        logger.info(f"正在通过网页请求知乎热榜: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        # 检查响应状态
        if response.status_code != 200:
            logger.warning(f"网页请求知乎热榜失败，状态码: {response.status_code}")
            return []
        
        # 保存HTML内容到文件，用于调试
        with open("zhihu_hot.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试多种选择器
        selectors = [
            '.HotList-item .HotItem-title',
            '.HotItem-content .HotItem-title',
            '.Card .ContentItem-title',
            '.HotItem .HotItem-title',
            'div[data-za-detail-view-path-module="HotItem"] h2'
        ]
        
        questions = []
        
        for selector in selectors:
            elements = soup.select(selector)
            logger.info(f"使用选择器 '{selector}' 找到 {len(elements)} 个元素")
            
            for element in elements:
                title = element.get_text().strip()
                # 只保留问号结尾的内容（问题）
                if "？" in title or "?" in title:
                    questions.append(title)
            
            # 如果已经找到问题，就不再尝试其他选择器
            if questions:
                break
        
        # 尝试从HTML中提取JSON数据
        if not questions:
            logger.info("尝试从HTML中提取JSON数据")
            json_data = re.search(r'<script id="js-initialData" type="text/json">(.*?)</script>', response.text)
            if json_data:
                try:
                    data = json.loads(json_data.group(1))
                    if 'initialState' in data and 'topstory' in data['initialState'] and 'hotList' in data['initialState']['topstory']:
                        hot_list = data['initialState']['topstory']['hotList']
                        for item in hot_list:
                            if 'target' in item and 'titleArea' in item['target'] and 'text' in item['target']['titleArea']:
                                title = item['target']['titleArea']['text']
                                if "？" in title or "?" in title:
                                    questions.append(title)
                except Exception as e:
                    logger.error(f"解析JSON数据时出错: {str(e)}")
        
        logger.info(f"通过网页成功获取 {len(questions)} 个知乎热榜问题")
        
        # 如果没有获取到足够的问题，使用备用问题
        if len(questions) < limit:
            logger.warning(f"从网页只获取到 {len(questions)} 个问题，将添加备用问题")
            questions.extend(get_fallback_questions(limit - len(questions)))
        
        return questions[:limit]
    except Exception as e:
        logger.error(f"通过网页获取知乎热榜问题时出错: {str(e)}")
        return []

def get_zhihu_hot_via_search(limit=10):
    """通过搜索页面获取热门问题"""
    try:
        # 知乎搜索热门页面
        url = "https://www.zhihu.com/search?type=question&q=热门"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        logger.info(f"正在通过搜索页面获取知乎热门问题: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"搜索页面请求失败，状态码: {response.status_code}")
            return []
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试多种选择器
        selectors = [
            '.QuestionItem-title',
            '.ContentItem-title',
            '.SearchResult-Card .ContentItem-title',
            'a[data-za-detail-view-element_name="Title"]'
        ]
        
        questions = []
        
        for selector in selectors:
            elements = soup.select(selector)
            logger.info(f"使用选择器 '{selector}' 找到 {len(elements)} 个元素")
            
            for element in elements:
                title = element.get_text().strip()
                # 只保留问号结尾的内容（问题）
                if "？" in title or "?" in title:
                    questions.append(title)
            
            # 如果已经找到问题，就不再尝试其他选择器
            if questions:
                break
        
        logger.info(f"通过搜索页面成功获取 {len(questions)} 个知乎热门问题")
        
        # 如果没有获取到足够的问题，使用备用问题
        if len(questions) < limit:
            logger.warning(f"从搜索页面只获取到 {len(questions)} 个问题，将添加备用问题")
            questions.extend(get_fallback_questions(limit - len(questions)))
        
        return questions[:limit]
    except Exception as e:
        logger.error(f"通过搜索页面获取知乎热门问题时出错: {str(e)}")
        return []

def get_fallback_questions(count=10):
    """
    当无法获取知乎热榜时，返回备用问题列表
    
    Args:
        count: 需要的问题数量
        
    Returns:
        list: 备用问题列表
    """
    fallback_questions = [
        "AI会取代程序员吗？", 
        "如何学习大模型？",
        "ChatGPT对教育行业有什么影响？",
        "2024年值得关注的科技趋势有哪些？",
        "如何平衡工作与生活？",
        "不会编程，直接用AI来写代码靠谱吗？",
        "人工智能会对就业市场产生什么影响？",
        "如何提高自己的学习效率？",
        "大模型时代，普通人如何保持竞争力？",
        "未来十年最有前景的行业是什么？",
        "如何培养自己的创造力？",
        "远程工作会成为未来的主流吗？",
        "如何有效地进行时间管理？",
        "元宇宙概念是炒作还是未来趋势？",
        "如何开始投资理财？"
    ]
    
    # 如果备用问题不够，就重复使用
    if count > len(fallback_questions):
        return fallback_questions
    
    # 随机选择指定数量的问题
    return random.sample(fallback_questions, count)

# 测试代码
if __name__ == "__main__":
    questions = get_zhihu_hot_questions()
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")