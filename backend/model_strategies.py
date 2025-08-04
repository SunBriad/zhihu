import os
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelStrategy(ABC):
    """模型策略抽象基类"""
    
    @abstractmethod
    def analyze_question(self, question: str, tone: str, length: str) -> str:
        """分析问题"""
        pass
    
    @abstractmethod
    def generate_answer(self, question: str, context: List[str], tone: str, word_count: str) -> str:
        """生成回答"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass

class FakeStrategy(ModelStrategy):
    """假模型策略（当所有模型都不可用时使用）"""
    
    def analyze_question(self, question: str, tone: str, length: str) -> str:
        """分析问题"""
        return f"这是一个关于'{question}'的问题分析。请使用{tone}的语气，{length}的长度回答。"
    
    def generate_answer(self, question: str, context: List[str], tone: str, word_count: str) -> str:
        """生成回答"""
        return f"""# 关于"{question}"的回答
        
很抱歉，当前没有可用的模型API密钥。请在环境变量中设置以下任一API密钥：
- ZHIPU_API_KEY (智谱AI)
- DEEPSEEK_API_KEY (DeepSeek)
- DASHSCOPE_API_KEY (阿里云通义千问)
- KIMI_API_KEY (Moonshot AI Kimi)
- OPENAI_API_KEY (OpenAI)

然后重新运行程序。
        """
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        # 返回1536维的全零向量
        return [[0.0] * 1536 for _ in texts]
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return True  # 假模型始终可用