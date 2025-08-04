from typing import Any, Dict, List, Optional
import logging
import os
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, model_validator

# 尝试导入zhipuai
try:
    import zhipuai
    zhipuai_available = True
except ImportError:
    zhipuai_available = False

# 配置日志
logger = logging.getLogger(__name__)

class ZhipuEmbeddings(BaseModel, Embeddings):
    """智谱AI文本嵌入模型。
    
    使用智谱AI的文本嵌入模型生成文本的向量表示。
    """
    
    api_key: Optional[str] = None
    """智谱AI API密钥"""
    
    model: str = "embedding-2"
    """使用的嵌入模型名称，默认为embedding-2"""
    
    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, data: Dict) -> Dict:
        """验证环境变量和依赖项。"""
        if not zhipuai_available:
            raise ImportError(
                "无法导入zhipuai包。请使用 `pip install zhipuai` 安装。"
            )
        
        api_key = data.get("api_key") or os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError(
                "智谱AI API密钥未设置。请设置ZHIPU_API_KEY环境变量或在初始化时提供api_key参数。"
            )
        
        data["api_key"] = api_key
        return data
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成多个文本的嵌入向量。
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            嵌入向量列表
        """
        try:
            # 初始化智谱AI客户端
            zhipuai.api_key = self.api_key
            
            # 批量处理文本，避免超出API限制
            embeddings = []
            batch_size = 16  # 每批处理的文本数量
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                try:
                    # 调用智谱AI嵌入API
                    response = zhipuai.model_api.invoke(
                        model=self.model,
                        prompt=batch_texts,
                        task_type="embeddings"
                    )
                    
                    if response.get("code") == 200:
                        # 提取嵌入向量
                        batch_embeddings = [item["embedding"] for item in response["data"]]
                        embeddings.extend(batch_embeddings)
                    else:
                        logger.error(f"嵌入API调用失败: {response.get('code')}, {response.get('msg')}")
                        # 返回零向量作为后备
                        for _ in batch_texts:
                            embeddings.append([0.0] * 1536)  # 假设向量维度为1536
                except Exception as e:
                    logger.error(f"处理批次 {i} 时出错: {str(e)}")
                    # 返回零向量作为后备
                    for _ in batch_texts:
                        embeddings.append([0.0] * 1536)  # 假设向量维度为1536
            
            return embeddings
            
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            # 返回零向量作为后备
            return [[0.0] * 1536 for _ in texts]  # 假设向量维度为1536
    
    def embed_query(self, text: str) -> List[float]:
        """生成单个查询文本的嵌入向量。
        
        Args:
            text: 要嵌入的查询文本
            
        Returns:
            嵌入向量
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else [0.0] * 1536  # 假设向量维度为1536
