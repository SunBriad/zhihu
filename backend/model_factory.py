import logging
from typing import Optional, Dict, Type
from .model_strategies import ModelStrategy
from .zhipu_strategy import ZhipuStrategy
from .deepseek_strategy import DeepSeekStrategy
from .qwen_strategy import QwenStrategy
from .kimi_strategy import KimiStrategy
from .openai_strategy import OpenAIStrategy

# 配置日志
logger = logging.getLogger(__name__)

class ModelFactory:
    """模型工厂类，用于创建和管理不同的模型策略"""
    
    def __init__(self):
        # 注册所有可用的模型策略
        self._strategies: Dict[str, Type[ModelStrategy]] = {
            "zhipu": ZhipuStrategy,
            "deepseek": DeepSeekStrategy,
            "qwen": QwenStrategy,
            "kimi": KimiStrategy,
            "openai": OpenAIStrategy
        }
        
        # 初始化策略实例缓存
        self._strategy_instances: Dict[str, ModelStrategy] = {}
        
        # 默认策略
        self._default_strategy_name = "zhipu"
    
    def get_strategy(self, strategy_name: Optional[str] = None) -> ModelStrategy:
        """
        获取指定名称的模型策略
        
        Args:
            strategy_name: 策略名称，如果为None则自动选择可用的策略
            
        Returns:
            ModelStrategy: 模型策略实例
            
        Raises:
            ValueError: 如果没有可用的策略
        """
        # 如果指定了策略名称，尝试获取该策略
        if strategy_name:
            return self._get_specific_strategy(strategy_name)
        
        # 否则，自动选择可用的策略
        return self._get_available_strategy()
    
    def _get_specific_strategy(self, strategy_name: str) -> ModelStrategy:
        """获取指定名称的策略"""
        # 如果是auto策略，则自动选择可用的策略
        if strategy_name == "auto":
            return self._get_available_strategy()
            
        # 检查策略名称是否有效
        if strategy_name not in self._strategies:
            raise ValueError(f"未知的模型策略: {strategy_name}")
        
        # 如果策略实例已经存在，直接返回
        if strategy_name in self._strategy_instances:
            strategy = self._strategy_instances[strategy_name]
            if strategy.is_available():
                return strategy
        
        # 创建新的策略实例
        strategy_class = self._strategies[strategy_name]
        strategy = strategy_class()
        
        # 检查策略是否可用
        if not strategy.is_available():
            raise ValueError(f"模型策略 {strategy_name} 不可用，请检查API密钥配置")
        
        # 缓存策略实例
        self._strategy_instances[strategy_name] = strategy
        return strategy
    
    def _get_available_strategy(self) -> ModelStrategy:
        """自动选择可用的策略"""
        # 首先尝试默认策略
        try:
            return self._get_specific_strategy(self._default_strategy_name)
        except ValueError:
            pass
        
        # 尝试所有注册的策略
        for strategy_name in self._strategies:
            try:
                return self._get_specific_strategy(strategy_name)
            except ValueError:
                continue
        
        # 如果没有可用的策略，抛出异常
        raise ValueError("没有可用的模型策略，请检查API密钥配置")
    
    def list_available_strategies(self) -> Dict[str, bool]:
        """
        列出所有策略及其可用状态
        
        Returns:
            Dict[str, bool]: 策略名称到可用状态的映射
        """
        result = {}
        
        for strategy_name in self._strategies:
            try:
                # 尝试获取策略实例
                if strategy_name in self._strategy_instances:
                    strategy = self._strategy_instances[strategy_name]
                else:
                    strategy_class = self._strategies[strategy_name]
                    strategy = strategy_class()
                    self._strategy_instances[strategy_name] = strategy
                
                # 检查策略是否可用
                result[strategy_name] = strategy.is_available()
            except Exception as e:
                logger.error(f"检查策略 {strategy_name} 可用性时出错: {str(e)}")
                result[strategy_name] = False
        
        return result

# 创建全局模型工厂实例
model_factory = ModelFactory()

def get_model_strategy(strategy_name: Optional[str] = None) -> ModelStrategy:
    """
    获取模型策略的便捷函数
    
    Args:
        strategy_name: 策略名称，如果为None则自动选择可用的策略
        
    Returns:
        ModelStrategy: 模型策略实例
    """
    return model_factory.get_strategy(strategy_name)