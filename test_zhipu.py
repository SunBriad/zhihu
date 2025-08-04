import os
import dotenv

# 加载.env文件
dotenv.load_dotenv()

# 从环境变量获取API密钥
api_key = os.environ.get("ZHIPU_API_KEY")
print(f"API密钥: {api_key[:5]}{'*' * 10 if api_key else '未设置'}")

try:
    # 使用zhipuai库
    import zhipuai
    
    # 初始化客户端
    zhipuai.api_key = api_key
    
    # 查看可用的方法和属性
    print("zhipuai库的属性和方法:")
    for attr in dir(zhipuai):
        if not attr.startswith('__'):
            print(f"- {attr}")
    
    # 尝试使用新版API
    try:
        # 尝试使用chat.completions接口
        client = zhipuai.ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": "你好，请用一句话介绍自己"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        print("\nAPI调用成功 (chat.completions)!")
        print("回复内容:", response.choices[0].message.content)
    except Exception as e1:
        print(f"\n使用chat.completions接口失败: {str(e1)}")
        
        # 尝试使用旧版API
        try:
            response = zhipuai.model_api.invoke(
                model="glm-4",
                prompt="你好，请用一句话介绍自己",
                top_p=0.7,
                temperature=0.7,
                max_tokens=100
            )
            print("\nAPI调用成功 (model_api)!")
            print("回复内容:", response)
        except Exception as e2:
            print(f"\n使用model_api接口失败: {str(e2)}")
            
            # 尝试其他可能的接口
            print("\n尝试查找其他可能的接口...")
            if hasattr(zhipuai, 'core'):
                print("发现core模块，属性和方法:")
                for attr in dir(zhipuai.core):
                    if not attr.startswith('__'):
                        print(f"- {attr}")
    
except ImportError as e:
    print(f"导入zhipuai库失败: {str(e)}")
except Exception as e:
    print(f"API调用失败: {str(e)}")