import os
import logging
from typing import List
from .model_strategies import ModelStrategy

# 配置日志
logger = logging.getLogger(__name__)

class OpenAIStrategy(ModelStrategy):
    """OpenAI模型策略"""
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """检查OpenAI API是否可用"""
        if not self.api_key or len(self.api_key) < 10:
            logger.warning("OpenAI API密钥未设置或无效")
            return False
        return True
    
    def analyze_question(self, question: str, tone: str, length: str) -> str:
        """使用OpenAI模型分析问题"""
        if not self.is_available():
            raise ValueError("OpenAI API不可用")
        
        try:
            from openai import OpenAI
            
            logger.info("使用OpenAI模型分析问题")
            client = OpenAI(api_key=self.api_key)
            
            # 构建提示词
            prompt_text = f"""请分析以下问题，并思考如何回答：
            
            问题：{question}
            
            要求：
            - 语气：{tone}
            - 回答长度：{length}
            
            请提供你的分析思路（不是回答本身）："""
            
            # 调用OpenAI模型
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "你是一位专业的知乎回答分析专家，擅长分析问题并提供思路。"},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            analysis = response.choices[0].message.content
            logger.info("OpenAI问题分析完成")
            return analysis
            
        except Exception as e:
            logger.error(f"使用OpenAI模型分析问题时出错: {str(e)}")
            raise
    
    def generate_answer(self, question: str, context: List[str], tone: str, word_count: str) -> str:
        """使用OpenAI模型生成回答"""
        if not self.is_available():
            raise ValueError("OpenAI API不可用")
        
        try:
            from openai import OpenAI
            
            logger.info("使用OpenAI模型生成回答")
            client = OpenAI(api_key=self.api_key)
            
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
            
            # 调用OpenAI模型
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "你是一位专业的知乎回答者，擅长生成高质量、有深度的回答。"},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            answer = response.choices[0].message.content
            logger.info("OpenAI回答生成完成")
            return answer
            
        except Exception as e:
            logger.error(f"使用OpenAI模型生成回答时出错: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        if not self.is_available():
            raise ValueError("OpenAI API不可用")
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            embeddings = []
            
            for text in texts:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"使用OpenAI获取嵌入向量时出错: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.available