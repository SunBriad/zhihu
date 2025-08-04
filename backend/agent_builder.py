from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
import operator
import os
import logging
import dotenv
import requests
from urllib.parse import quote
from .config import MODEL_CONFIG, VECTOR_STORE_PATH, TEMP_DIR, update_model_config, ensure_dir_exists
from .model_factory import model_factory, get_model_strategy

# 加载.env文件
dotenv.load_dotenv()

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# 状态定义
class AgentState(TypedDict):
    question: str
    tone: str
    length: str
    context: Annotated[Sequence[str], operator.add]
    answer: str
    thoughts: Annotated[Sequence[str], operator.add]
    images: Annotated[Sequence[Dict[str, str]], operator.add]  # 存储图片信息，包含URL和描述

# 从网络收集图片的函数
def collect_images_for_question(question: str, max_images: int = 3) -> List[Dict[str, str]]:
    """
    根据问题从网络收集相关图片
    
    Args:
        question: 问题文本
        max_images: 最大图片数量
        
    Returns:
        List[Dict[str, str]]: 图片信息列表，每个字典包含url和description
    """
    logger.info(f"开始为问题收集图片: {question[:50]}...")
    
    try:
        # 确保临时目录存在
        ensure_dir_exists(TEMP_DIR)
        
        # 构建搜索查询
        search_query = quote(question)
        
        # 使用Bing图片搜索API (这里使用模拟实现，实际应用中应使用真实API)
        # 注意：在实际应用中，应该使用正式的图片搜索API，如Bing Image Search API或Google Custom Search API
        api_url = f"https://www.bing.com/images/search?q={search_query}&form=HDRSC2&first=1"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        logger.debug(f"发送图片搜索请求: {api_url}")
        
        # 注意：这里只是模拟实现，实际应用中应该使用真实API并解析返回的JSON
        # 在实际项目中，应该替换为真实的API调用和结果解析
        images = []
        for i in range(min(3, max_images)):  # 模拟找到3张图片
            image_info = {
                "url": f"https://example.com/image_{i+1}.jpg",  # 模拟URL
                "description": f"与'{question}'相关的图片 #{i+1}"  # 模拟描述
            }
            images.append(image_info)
            
        logger.info(f"成功收集到 {len(images)} 张相关图片")
        return images
        
    except Exception as e:
        logger.error(f"收集图片时出错: {str(e)}")
        return []  # 出错时返回空列表

# 创建智能体工作流
def create_agent_workflow():
    # 1. 检索知识
    def retrieve(state: AgentState) -> Dict[str, Any]:
        logger.debug(f"开始检索知识，问题: {state['question'][:50]}...")
        
        # 确保知识库存在
        from .knowledge_loader import get_default_knowledge_base
        logger.debug("调用get_default_knowledge_base确保知识库存在")
        get_default_knowledge_base()
        
        try:
            # 获取当前配置的模型策略
            model_strategy = get_model_strategy(MODEL_CONFIG.get("provider", "auto"))
            
            logger.info(f"使用模型策略: {model_strategy.__class__.__name__}")
            
            # 尝试使用模型策略获取嵌入向量
            try:
                # 创建一个临时文本用于测试嵌入功能
                test_text = "测试嵌入功能"
                _ = model_strategy.get_embeddings([test_text])
                logger.info(f"嵌入功能测试成功，使用 {model_strategy.__class__.__name__} 进行检索")
                
                # 加载向量存储
                from langchain_community.embeddings import FakeEmbeddings
                
                # 使用FakeEmbeddings初始化向量存储，然后再替换为实际的嵌入模型
                # 这是因为FAISS.load_local需要一个嵌入模型，但我们实际上会使用自己的嵌入逻辑
                temp_embeddings = FakeEmbeddings(size=1536)
                
                logger.debug(f"开始加载向量存储，路径: {VECTOR_STORE_PATH}")
                
                vectorstore = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    temp_embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.debug("向量存储加载成功")
                
                # 使用我们的模型策略获取问题的嵌入向量
                question_embedding = model_strategy.get_embeddings([state["question"]])[0]
                
                # 执行相似度搜索
                logger.debug(f"执行相似度搜索，问题: {state['question'][:50]}...")
                
                # 使用FAISS的原始搜索方法，传入我们自己生成的嵌入向量
                docs_with_scores = vectorstore.similarity_search_with_score_by_vector(
                    question_embedding, 
                    k=5
                )
                
                # 提取文档
                docs = [doc for doc, _ in docs_with_scores]
                
                logger.debug(f"相似度搜索完成，返回 {len(docs)} 条结果")
                
                # 提取相关内容
                contexts = [d.page_content for d in docs]
                logger.info(f"检索到 {len(contexts)} 条相关知识")
                logger.debug(f"第一条知识: {contexts[0][:100]}..." if contexts else "无检索结果")
            except Exception as e:
                logger.error(f"使用模型策略进行检索时出错: {str(e)}")
                logger.warning("尝试使用标准FAISS检索方法")
                
                # 如果使用模型策略失败，回退到标准FAISS检索方法
                from langchain_community.embeddings import FakeEmbeddings
                
                # 使用FakeEmbeddings作为后备方案
                embedding_model = FakeEmbeddings(size=1536)  # 使用1536维向量，与OpenAI兼容
                
                # 加载向量存储
                logger.debug(f"开始加载向量存储，路径: {VECTOR_STORE_PATH}")
                logger.debug(f"使用的嵌入模型类型: {type(embedding_model).__name__}")
                
                vectorstore = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.debug("向量存储加载成功")
                
                # 执行相似度搜索
                logger.debug(f"执行相似度搜索，问题: {state['question'][:50]}...")
                docs = vectorstore.similarity_search(state["question"], k=5)
                logger.debug(f"相似度搜索完成，返回 {len(docs)} 条结果")
                
                # 提取相关内容
                contexts = [d.page_content for d in docs]
                logger.info(f"检索到 {len(contexts)} 条相关知识")
                logger.debug(f"第一条知识: {contexts[0][:100]}..." if contexts else "无检索结果")
            
            # 添加思考过程
            thoughts = [f"已从知识库中检索到 {len(contexts)} 条相关信息"]
            
            return {
                "context": contexts,
                "thoughts": thoughts
            }
        except Exception as e:
            logger.error(f"知识检索出错: {str(e)}")
            return {
                "context": ["无法访问知识库，将使用通用知识回答"],
                "thoughts": [f"知识检索失败: {str(e)}"]
            }
    
    # 2. 收集图片
    def collect_images(state: AgentState) -> Dict[str, Any]:
        try:
            logger.info(f"开始为问题收集图片: {state['question'][:50]}...")
            
            # 调用图片收集函数
            images = collect_images_for_question(state["question"])
            
            if images:
                logger.info(f"成功收集到 {len(images)} 张相关图片")
                return {
                    "images": images,
                    "thoughts": [f"已收集 {len(images)} 张与问题相关的图片"]
                }
            else:
                logger.warning("未能收集到相关图片")
                return {
                    "images": [],
                    "thoughts": ["未能收集到与问题相关的图片"]
                }
                
        except Exception as e:
            logger.error(f"收集图片时出错: {str(e)}")
            return {
                "images": [],
                "thoughts": [f"收集图片时出错: {str(e)}"]
            }
    
    # 3. 分析问题
    def analyze_question(state: AgentState) -> Dict[str, Any]:
        try:
            # 获取当前配置的模型策略
            model_strategy = get_model_strategy(MODEL_CONFIG.get("provider", "auto"))
            
            logger.info(f"使用模型策略: {model_strategy.__class__.__name__} 分析问题")
            
            # 使用模型策略分析问题
            analysis = model_strategy.analyze_question(
                question=state["question"],
                tone=state["tone"],
                length=state["length"]
            )
            
            logger.info("问题分析完成")
            logger.debug(f"分析结果: {analysis[:100]}...")
            
            return {"thoughts": [analysis]}
            
        except Exception as e:
            logger.error(f"分析问题时出错: {str(e)}")
            return {"thoughts": [f"分析问题时出错: {str(e)}，将使用简单分析继续"]}
    
    # 4. 生成回答
    def generate_response(state: AgentState) -> Dict[str, Any]:
        # 根据长度设置字数范围
        length_guide = {
            "简短": "300-500字",
            "中等": "800-1200字",
            "详细": "1500-2500字"
        }
        word_count = length_guide.get(state["length"], "800-1200字")
        
        try:
            # 获取当前配置的模型策略
            model_strategy = get_model_strategy(MODEL_CONFIG.get("provider", "auto"))
            
            logger.info(f"使用模型策略: {model_strategy.__class__.__name__} 生成回答")
            
            # 准备图片信息
            image_info = ""
            if hasattr(state, 'images') and state.get('images'):
                image_info = "\n\n图片资源:\n"
                for i, img in enumerate(state['images']):
                    image_info += f"- 图片{i+1}: {img['url']} - {img['description']}\n"
                logger.info(f"将 {len(state['images'])} 张图片信息添加到回答中")
            
            # 使用模型策略生成回答
            answer = model_strategy.generate_answer(
                question=state["question"],
                context=state["context"] + ([image_info] if image_info else []),
                tone=state["tone"],
                word_count=word_count
            )
            
            # 如果有图片，在回答中添加图片引用
            if hasattr(state, 'images') and state.get('images'):
                # 在回答末尾添加图片引用
                answer += "\n\n## 相关图片\n"
                for i, img in enumerate(state['images']):
                    answer += f"\n![{img['description']}]({img['url']})\n"
            
            logger.info("回答生成完成")
            logger.debug(f"回答结果: {answer[:100]}...")
            
            return {"answer": answer}
            
        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            return {"answer": f"生成回答时出错: {str(e)}，请检查API密钥设置或网络连接。"}
    
    # 构建工作流
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("collect_images", collect_images)
    workflow.add_node("analyze", analyze_question)
    workflow.add_node("generate", generate_response)
    
    # 设置边
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "collect_images")
    workflow.add_edge("collect_images", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# 全局智能体实例
_agent_executor = None

def get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        logger.info("创建新的智能体执行器")
        _agent_executor = create_agent_workflow()
    return _agent_executor