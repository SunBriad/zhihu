from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import FakeEmbeddings
import os
import tempfile
import logging
from .config import MODEL_CONFIG, VECTOR_STORE_PATH, TEMP_DIR, ensure_dir_exists
from .ali_embeddings import AliTextEmbeddings
from .zhipu_embeddings import ZhipuEmbeddings

# 获取日志记录器
logger = logging.getLogger(__name__)

def load_knowledge_base(files):
    """加载知识库文件并创建向量存储"""
    ensure_dir_exists(TEMP_DIR)
    ensure_dir_exists(VECTOR_STORE_PATH)
    
    documents = []
    temp_files = []
    
    try:
        for file in files:
            # 保存上传文件到临时目录
            file_path = os.path.join(TEMP_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            temp_files.append(file_path)
            
            logger.info(f"处理文件: {file.name}")
            
            # 根据文件类型选择合适的加载器
            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                logger.info(f"使用 PyPDFLoader 加载 {file.name}")
            elif file.name.lower().endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
                logger.info(f"使用 UnstructuredMarkdownLoader 加载 {file.name}")
            else:
                # 尝试多种编码加载文本文件
                try:
                    # 首先尝试 UTF-8 编码
                    loader = TextLoader(file_path, encoding="utf-8")
                    logger.info(f"使用 TextLoader 加载 {file.name}，编码：utf-8")
                except Exception as e:
                    logger.warning(f"UTF-8 编码加载失败: {str(e)}，尝试其他编码")
                    try:
                        # 尝试 GBK 编码
                        loader = TextLoader(file_path, encoding="gbk")
                        logger.info(f"使用 TextLoader 加载 {file.name}，编码：gbk")
                    except Exception as e:
                        logger.warning(f"GBK 编码加载失败: {str(e)}，尝试 latin-1 编码")
                        # 最后尝试 latin-1 编码，它可以加载任何字节序列
                        loader = TextLoader(file_path, encoding="latin-1")
                        logger.info(f"使用 TextLoader 加载 {file.name}，编码：latin-1")
                
            documents.extend(loader.load())
        
        if not documents:
            logger.warning("没有加载到任何文档")
            return False
            
        logger.info(f"成功加载 {len(documents)} 个文档")
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"文档分割为 {len(splits)} 个片段")
        
        # 根据配置选择嵌入模型
        try:
            # 首先尝试使用智谱AI嵌入模型
            if os.environ.get("ZHIPU_API_KEY") and len(os.environ.get("ZHIPU_API_KEY")) > 10:
                logger.info("使用智谱AI嵌入模型")
                embedding_model = ZhipuEmbeddings()
            # 然后尝试使用OpenAI嵌入模型
            elif MODEL_CONFIG.get("api_key") and MODEL_CONFIG["api_key"].startswith("sk-") and len(MODEL_CONFIG["api_key"]) > 20:
                logger.info("使用OpenAI嵌入模型")
                embedding_model = OpenAIEmbeddings(api_key=MODEL_CONFIG["api_key"])
            # 如果没有OpenAI API密钥，尝试使用阿里云嵌入模型
            elif os.environ.get("DASHSCOPE_API_KEY") and len(os.environ.get("DASHSCOPE_API_KEY")) > 10:
                logger.info("使用阿里云嵌入模型")
                embedding_model = AliTextEmbeddings()
            else:
                logger.warning("未找到有效的API密钥，使用FakeEmbeddings作为后备方案")
                embedding_model = FakeEmbeddings(size=1536)  # 使用1536维向量，与OpenAI兼容
                
            # 创建向量库
            vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=embedding_model
            )
        except Exception as e:
            logger.error(f"创建嵌入模型时出错: {str(e)}")
            raise
        vectorstore.save_local(VECTOR_STORE_PATH)
        logger.info(f"向量库已保存到 {VECTOR_STORE_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"加载知识库时出错: {str(e)}")
        raise
        
    finally:
        # 清理临时文件
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"已删除临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"删除临时文件 {file_path} 时出错: {str(e)}")

def get_default_knowledge_base():
    """检查是否存在默认知识库，如果不存在则创建一个简单的默认知识库"""
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        logger.info("未找到现有知识库，创建默认知识库")
        
        # 创建一个简单的默认文档
        ensure_dir_exists(TEMP_DIR)
        default_doc_path = os.path.join(TEMP_DIR, "default_knowledge.txt")
        
        with open(default_doc_path, "w", encoding="utf-8") as f:
            f.write("""
            知乎是中国知名的问答社区，用户可以在平台上提问、回答问题，分享知识和经验。
            回答知乎问题时，应当注重逻辑性和专业性，提供有价值的信息和见解。
            好的知乎回答通常包含个人经验、专业知识和数据支持，能够全面解答提问者的疑惑。
            在知乎上，清晰的结构、适当的例证和真诚的态度往往能获得更多的认可。
            """)
            
        # 加载默认文档
        loader = TextLoader(default_doc_path)
        documents = loader.load()
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 根据配置选择嵌入模型
        try:
            # 首先尝试使用智谱AI嵌入模型
            if os.environ.get("ZHIPU_API_KEY") and len(os.environ.get("ZHIPU_API_KEY")) > 10:
                logger.info("使用智谱AI嵌入模型")
                embedding_model = ZhipuEmbeddings()
            # 然后尝试使用OpenAI嵌入模型
            elif MODEL_CONFIG.get("api_key") and MODEL_CONFIG["api_key"].startswith("sk-") and len(MODEL_CONFIG["api_key"]) > 20:
                logger.info("使用OpenAI嵌入模型")
                embedding_model = OpenAIEmbeddings(api_key=MODEL_CONFIG["api_key"])
            # 如果没有OpenAI API密钥，尝试使用阿里云嵌入模型
            elif os.environ.get("DASHSCOPE_API_KEY") and len(os.environ.get("DASHSCOPE_API_KEY")) > 10:
                logger.info("使用阿里云嵌入模型")
                embedding_model = AliTextEmbeddings()
            else:
                logger.warning("未找到有效的API密钥，使用FakeEmbeddings作为后备方案")
                embedding_model = FakeEmbeddings(size=1536)  # 使用1536维向量，与OpenAI兼容
                
            # 创建向量库
            vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=embedding_model
            )
        except Exception as e:
            logger.error(f"创建默认知识库时出错: {str(e)}")
            raise
        vectorstore.save_local(VECTOR_STORE_PATH)
        logger.info("默认知识库已创建")
        
        # 清理临时文件
        os.remove(default_doc_path)
        
    return True