import os
import logging
from typing import List
from .model_strategies import ModelStrategy

# 配置日志
logger = logging.getLogger(__name__)

class KimiStrategy(ModelStrategy):
    """Kimi模型策略"""
    
    def __init__(self):
        self.api_key = os.environ.get("KIMI_API_KEY")
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """检查Kimi API是否可用"""
        if not self.api_key or len(self.api_key) < 10:
            logger.warning("Kimi API密钥未设置或无效")
            return False
        return True
    
    def analyze_question(self, question: str, tone: str, length: str) -> str:
        """使用Kimi模型分析问题"""
        if not self.is_available():
            raise ValueError("Kimi API不可用")
        
        try:
            import requests
            import json
            
            logger.info("使用Kimi模型分析问题")
            
            # 构建提示词
            prompt_text = f"""请分析以下问题，并思考如何回答：
            
            问题：{question}
            
            要求：
            - 语气：{tone}
            - 回答长度：{length}
            
            请提供你的分析思路（不是回答本身）："""
            
            # 调用Kimi API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "moonshot-v1-8k",
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                logger.info("Kimi问题分析完成")
                return analysis
            else:
                logger.error(f"Kimi API响应异常: {response.text}")
                raise ValueError(f"Kimi API响应异常: {response.status_code}")
            
        except Exception as e:
            logger.error(f"使用Kimi模型分析问题时出错: {str(e)}")
            raise
    
    def generate_answer(self, question: str, context: List[str], tone: str, word_count: str) -> str:
        """使用Kimi模型生成回答"""
        if not self.is_available():
            raise ValueError("Kimi API不可用")
        
        try:
            import requests
            import json
            
            logger.info("使用Kimi模型生成回答")
            
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
            
            # 调用Kimi API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "moonshot-v1-8k",
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info("Kimi回答生成完成")
                return answer
            else:
                logger.error(f"Kimi API响应异常: {response.text}")
                raise ValueError(f"Kimi API响应异常: {response.status_code}")
            
        except Exception as e:
            logger.error(f"使用Kimi模型生成回答时出错: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        if not self.is_available():
            raise ValueError("Kimi API不可用")
        
        try:
            import requests
            import json
            
            embeddings = []
            
            for text in texts:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "embedding-2",
                    "input": text
                }
                
                response = requests.post(
                    "https://api.moonshot.cn/v1/embeddings",
                    headers=headers,
                    data=json.dumps(data)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result["data"][0]["embedding"]
                    embeddings.append(embedding)
                else:
                    logger.error(f"Kimi Embedding API响应异常: {response.text}")
                    raise ValueError(f"Kimi Embedding API响应异常: {response.status_code}")
            
            return embeddings
        except Exception as e:
            logger.error(f"使用Kimi获取嵌入向量时出错: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.available