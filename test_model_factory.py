import os
import logging
import dotenv
from backend.model_factory import model_factory

# 加载环境变量
dotenv.load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_factory():
    """测试模型工厂"""
    logger.info("开始测试模型工厂")
    
    # 使用模型工厂
    
    # 测试自动选择策略
    logger.info("测试自动选择策略")
    auto_strategy = model_factory.get_strategy("auto")
    logger.info(f"自动选择策略: {auto_strategy.__class__.__name__}")
    
    # 测试各个模型策略
    providers = ["zhipu", "deepseek", "qwen", "kimi", "openai"]
    
    for provider in providers:
        logger.info(f"测试 {provider} 策略")
        try:
            # 获取策略实例
            strategy_class = model_factory._strategies[provider]
            strategy = strategy_class()
            
            logger.info(f"{provider} 策略: {strategy.__class__.__name__}")
            logger.info(f"{provider} 策略可用性: {strategy.is_available()}")
        
            # 只有当策略可用时才进行测试
            if strategy.is_available():
                try:
                    # 测试分析问题
                    logger.info(f"测试 {provider} 分析问题")
                    question = "人工智能对未来的影响是什么？"
                    analysis = strategy.analyze_question(question, "专业严谨", "中等")
                    logger.info(f"{provider} 分析结果: {analysis[:100]}...")
                    
                    # 测试生成回答
                    logger.info(f"测试 {provider} 生成回答")
                    context = ["人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。", 
                              "人工智能技术包括机器学习、深度学习、自然语言处理等。"]
                    answer = strategy.generate_answer(question, context, "专业严谨", "300-500字")
                    logger.info(f"{provider} 回答结果: {answer[:100]}...")
                    
                    # 测试获取嵌入向量
                    logger.info(f"测试 {provider} 获取嵌入向量")
                    texts = ["这是一个测试文本，用于测试嵌入功能。"]
                    embeddings = strategy.get_embeddings(texts)
                    logger.info(f"{provider} 嵌入向量维度: {len(embeddings[0])}")
                    
                    logger.info(f"{provider} 策略测试成功")
                except Exception as e:
                    logger.error(f"{provider} 策略测试失败: {str(e)}")
            else:
                logger.warning(f"{provider} 策略不可用，跳过测试")
        except Exception as e:
            logger.error(f"初始化 {provider} 策略时出错: {str(e)}")
    
    logger.info("模型工厂测试完成")

if __name__ == "__main__":
    test_model_factory()