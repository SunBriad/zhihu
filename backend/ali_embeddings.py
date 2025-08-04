import os
import logging
from typing import List, Optional, Any, Dict

# 配置日志
logger = logging.getLogger(__name__)

class AliTextEmbeddings:
    """阿里云文本嵌入模型封装类"""
    
    def __init__(self, model_name: str = "text-embedding-v2"):
        """
        初始化阿里云文本嵌入模型
        
        Args:
            model_name: 模型名称，默认为"text-embedding-v2"
        """
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
        
        self.model_name = model_name
    
    def embed_query(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        try:
            import dashscope
            from dashscope import TextEmbedding
            
            dashscope.api_key = self.api_key
            
            response = TextEmbedding.call(
                model=self.model_name,
                input=text
            )
            
            if response.status_code == 200:
                # 根据阿里云API的实际响应格式进行解析
                if hasattr(response.output, 'embeddings') and len(response.output.embeddings) > 0:
                    embedding = response.output.embeddings[0].embedding
                    return embedding
                elif isinstance(response.output, dict) and 'embeddings' in response.output:
                    embedding = response.output['embeddings'][0]['embedding']
                    return embedding
                else:
                    # 尝试直接从响应中获取嵌入向量
                    logger.debug(f"阿里云嵌入API响应格式: {type(response.output)}")
                    logger.debug(f"阿里云嵌入API响应内容: {response.output}")
                    
                    # 如果是字典类型，尝试从不同的字段获取嵌入向量
                    if isinstance(response.output, dict):
                        if 'embedding' in response.output:
                            return response.output['embedding']
                        elif 'data' in response.output and len(response.output['data']) > 0:
                            if 'embedding' in response.output['data'][0]:
                                return response.output['data'][0]['embedding']
                    
                    # 如果无法解析，返回一个空向量
                    logger.error(f"无法从阿里云嵌入API响应中解析嵌入向量: {response.output}")
                    return [0.0] * 1536  # 返回1536维的零向量作为后备方案
            else:
                logger.error(f"阿里云嵌入API响应异常: {response.message}")
                raise ValueError(f"阿里云嵌入API响应异常: {response.message}")
        
        except Exception as e:
            logger.error(f"获取文本嵌入向量时出错: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        获取多个文本的嵌入向量
        
        Args:
            texts: 输入文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []
        
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings