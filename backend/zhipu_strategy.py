import os
import logging
from typing import List
from .model_strategies import ModelStrategy

# 配置日志
logger = logging.getLogger(__name__)

class ZhipuStrategy(ModelStrategy):
    """智谱AI模型策略"""
    
    def __init__(self):
        self.api_key = os.environ.get("ZHIPU_API_KEY")
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """检查智谱AI API是否可用"""
        if not self.api_key or len(self.api_key) < 10:
            logger.warning("智谱AI API密钥未设置或无效")
            return False
        return True
    
    def analyze_question(self, question: str, tone: str, length: str) -> str:
        """使用智谱AI的GLM-4模型分析问题"""
        if not self.is_available():
            raise ValueError("智谱AI API不可用")
        
        try:
            import zhipuai
            
            logger.info("使用智谱AI的GLM-4模型分析问题")
            
            # 构建提示词
            prompt_text = f"""请分析以下问题，并思考如何回答：
            
            问题：{question}
            
            要求：
            - 语气：{tone}
            - 回答长度：{length}
            
            请提供你的分析思路（不是回答本身）："""
            
            # 调用智谱AI的GLM-4模型
            logger.debug(f"开始调用智谱AI的GLM-4模型进行问题分析")
            logger.debug(f"API密钥前5位: {self.api_key[:5] if self.api_key else '未设置'}")
            logger.debug(f"提示词: {prompt_text[:100]}...")
            
            client = zhipuai.ZhipuAI(api_key=self.api_key)
            logger.debug("已创建ZhipuAI客户端")
            
            response = client.chat.completions.create(
                model="glm-4",  # 使用GLM-4模型
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            # 处理响应
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    analysis = response.choices[0].message.content
                    logger.info("智谱AI问题分析完成")
                    return analysis
            
            logger.error("智谱AI响应格式异常")
            raise ValueError("智谱AI响应格式异常")
            
        except Exception as e:
            logger.error(f"使用智谱AI模型分析问题时出错: {str(e)}")
            raise
    
    def generate_answer(self, question: str, context: List[str], tone: str, word_count: str) -> str:
        """使用智谱AI的GLM-4模型生成回答"""
        if not self.is_available():
            raise ValueError("智谱AI API不可用")
        
        try:
            import zhipuai
            
            logger.info("使用智谱AI的GLM-4模型生成回答")
            
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
            
            # 调用智谱AI的GLM-4模型
            logger.debug(f"开始调用智谱AI的GLM-4模型生成回答")
            logger.debug(f"API密钥前5位: {self.api_key[:5] if self.api_key else '未设置'}")
            logger.debug(f"提示词: {prompt_text[:100]}...")
            
            client = zhipuai.ZhipuAI(api_key=self.api_key)
            logger.debug("已创建ZhipuAI客户端")
            
            response = client.chat.completions.create(
                model="glm-4",  # 使用GLM-4模型
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            # 处理响应
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    answer = response.choices[0].message.content
                    logger.info("智谱AI回答生成完成")
                    return answer
            
            logger.error("智谱AI响应格式异常")
            raise ValueError("智谱AI响应格式异常")
            
        except Exception as e:
            logger.error(f"使用智谱AI模型生成回答时出错: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        if not self.is_available():
            raise ValueError("智谱AI API不可用")
        
        try:
            from .zhipu_embeddings import ZhipuEmbeddings
            
            embedding_model = ZhipuEmbeddings()
            embeddings = []
            
            for text in texts:
                embedding = embedding_model.embed_query(text)
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"使用智谱AI获取嵌入向量时出错: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.available