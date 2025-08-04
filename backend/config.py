import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局配置
MODEL_CONFIG = {
    "provider": "auto",  # 可选值: "auto", "zhipu", "deepseek", "qwen", "kimi", "openai"
    "analysis_model": "gpt-3.5-turbo",
    "generation_model": "gpt-4-turbo",
    "api_key": os.environ.get("OPENAI_API_KEY", "")
}

# 支持的模型提供商
SUPPORTED_PROVIDERS = {
    "auto": {
        "name": "自动选择",
        "description": "自动选择可用的模型提供商",
        "env_var": None
    },
    "zhipu": {
        "name": "智谱AI",
        "description": "智谱AI的GLM-4模型",
        "env_var": "ZHIPU_API_KEY"
    },
    "deepseek": {
        "name": "DeepSeek",
        "description": "DeepSeek-chat模型",
        "env_var": "DEEPSEEK_API_KEY"
    },
    "qwen": {
        "name": "通义千问",
        "description": "阿里云通义千问模型",
        "env_var": "DASHSCOPE_API_KEY"
    },
    "kimi": {
        "name": "Kimi",
        "description": "Moonshot AI的Kimi模型",
        "env_var": "KIMI_API_KEY"
    },
    "openai": {
        "name": "OpenAI",
        "description": "OpenAI的GPT模型",
        "env_var": "OPENAI_API_KEY"
    }
}

def update_model_config(api_key=None, analysis_model=None, generation_model=None, provider=None):
    """更新模型配置"""
    global MODEL_CONFIG
    if api_key:
        MODEL_CONFIG["api_key"] = api_key
    if analysis_model:
        MODEL_CONFIG["analysis_model"] = analysis_model
    if generation_model:
        MODEL_CONFIG["generation_model"] = generation_model
    if provider:
        MODEL_CONFIG["provider"] = provider
    
    logger.info(f"模型配置已更新: 提供商={MODEL_CONFIG['provider']}")
    
    # 如果提供商是特定的中国大模型，将API密钥设置到对应的环境变量中
    if provider and provider != "auto" and provider != "openai":
        env_var = SUPPORTED_PROVIDERS.get(provider, {}).get("env_var")
        if env_var and api_key:
            os.environ[env_var] = api_key
            logger.info(f"已将API密钥设置到环境变量 {env_var}")

# 向量存储路径
VECTOR_STORE_PATH = "backend/vector_store/"
TEMP_DIR = "backend/temp/"

def ensure_dir_exists(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"创建目录: {dir_path}")