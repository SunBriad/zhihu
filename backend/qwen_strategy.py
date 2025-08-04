import os
import logging
from typing import List
from .model_strategies import ModelStrategy

# 配置日志
logger = logging.getLogger(__name__)

class QwenStrategy(ModelStrategy):
    """阿里云通义千问模型策略"""
    
    def __init__(self):
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """检查阿里云API是否可用"""
        if not self.api_key or len(self.api_key) < 10:
            logger.warning("阿里云API密钥未设置或无效")
            return False
        return True
    
    def analyze_question(self, question: str, tone: str, length: str) -> str:
        """使用阿里云通义千问模型分析问题"""
        if not self.is_available():
            raise ValueError("阿里云API不可用")
        
        try:
            from http import HTTPStatus
            import dashscope
            
            logger.info("使用阿里云通义千问模型分析问题")
            dashscope.api_key = self.api_key
            
            # 构建提示词
            prompt_text = f"""请分析以下问题，并思考如何回答：
            
            问题：{question}
            
            要求：
            - 语气：{tone}
            - 回答长度：{length}
            
            请提供你的分析思路（不是回答本身）："""
            
            # 调用阿里云通义千问模型
            from dashscope import Generation
            
            response = Generation.call(
                model='qwen-max',
                prompt=prompt_text,
                temperature=0.7,
                max_tokens=1024
            )
            
            if response.status_code == HTTPStatus.OK:
                analysis = response.output.text
                logger.info("阿里云通义千问问题分析完成")
                return analysis
            else:
                logger.error(f"阿里云通义千问API响应异常: {response.message}")
                raise ValueError(f"阿里云通义千问API响应异常: {response.message}")
            
        except Exception as e:
            logger.error(f"使用阿里云通义千问模型分析问题时出错: {str(e)}")
            raise
    
    def generate_answer(self, question: str, context: List[str], tone: str, word_count: str) -> str:
        """使用阿里云通义千问模型生成回答"""
        if not self.is_available():
            raise ValueError("阿里云API不可用")
        
        try:
            from http import HTTPStatus
            import dashscope
            
            logger.info("使用阿里云通义千问模型生成回答")
            dashscope.api_key = self.api_key
            
            # 构建提示词
            prompt_text = f"""你是一位专业的知乎回答者，请根据以下信息生成一篇高质量的知乎回答：
            
            问题：{question}
            
            参考知识：
            {"\n\n".join(context)}
            
            要求：
            1. 语气风格：{tone}
            2. 回答长度：{word_count}
            3. 结构清晰，有逻辑性，包含适当的小标题
            4. 内容真实可靠，避免虚构信息
            5. 如果知识库中没有相关信息，可以使用你的通用知识
            6. 适当引用数据或案例增加可信度
            7. 回答应当有个人见解，不要过于平淡
            8. 使用markdown格式美化回答
            
            你的回答："""
            
            # 调用阿里云通义千问模型
            from dashscope import Generation
            
            response = Generation.call(
                model='qwen-max',
                prompt=prompt_text,
                temperature=0.7,
                max_tokens=2048
            )
            
            if response.status_code == HTTPStatus.OK:
                answer = response.output.text
                logger.info("阿里云通义千问回答生成完成")
                return answer
            else:
                logger.error(f"阿里云通义千问API响应异常: {response.message}")
                raise ValueError(f"阿里云通义千问API响应异常: {response.message}")
            
        except Exception as e:
            logger.error(f"使用阿里云通义千问模型生成回答时出错: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        if not self.is_available():
            raise ValueError("阿里云API不可用")
        
        try:
            from .ali_embeddings import AliTextEmbeddings
            
            embedding_model = AliTextEmbeddings()
            embeddings = []
            
            for text in texts:
                embedding = embedding_model.embed_query(text)
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"使用阿里云获取嵌入向量时出错: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.available