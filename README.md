# 知乎热榜AI助手

这是一个基于大语言模型的知乎回答生成工具，支持多种中国大模型，包括智谱AI、DeepSeek、阿里云通义千问、Moonshot AI的Kimi和OpenAI。

## 功能特点

- 支持多种中国大模型：智谱AI(GLM-4)、DeepSeek、阿里云通义千问、Kimi和OpenAI
- 自动选择可用的模型策略
- 支持上传PDF、Word、TXT等多种格式的知识文档
- 基于知识文档生成高质量的知乎回答
- 支持自定义回答的语气和长度
- 支持一键发布到知乎（需要配置知乎Cookie）
- 支持获取知乎热榜问题

## 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/zhihu.git
cd zhihu
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
复制`.env.example`文件为`.env`，并填写相应的API密钥：
```bash
cp .env.example .env
```

编辑`.env`文件，填写您的API密钥：
```
# OpenAI API密钥（使用OpenAI嵌入模型时必填）
OPENAI_API_KEY=sk-your-openai-api-key

# 阿里云API密钥（使用阿里云嵌入模型时必填）
DASHSCOPE_API_KEY=sk-your-dashscope-api-key

# 智谱AI API密钥（使用智谱AI嵌入模型时必填）
ZHIPU_API_KEY=your-zhipu-api-key

# 知乎Cookie路径（可选，默认为cookies/zhihu_cookies.json）
ZHIHU_COOKIE_PATH=cookies/zhihu_cookies.json
```

## 使用方法

1. 运行应用
```bash
python run.py
```

2. 在浏览器中打开应用（默认地址：http://localhost:8501）

3. 上传知识文档（支持PDF、Word、TXT等格式）

4. 输入问题，选择模型提供商、语气和回答长度

5. 点击"生成回答"按钮

6. 查看生成的回答，可以一键发布到知乎（需要配置知乎Cookie）

## 配置知乎Cookie

1. 登录知乎网页版

2. 使用浏览器开发者工具导出Cookie

3. 将Cookie保存为JSON格式，放在`cookies/zhihu_cookies.json`目录下

## 项目结构

```
zhihu/
├── backend/                # 后端代码
│   ├── agent_builder.py    # 代理构建器
│   ├── ali_embeddings.py   # 阿里云嵌入向量实现
│   ├── config.py           # 配置文件
│   ├── deepseek_strategy.py # DeepSeek模型策略
│   ├── kimi_strategy.py    # Kimi模型策略
│   ├── knowledge_loader.py # 知识加载器
│   ├── model_factory.py    # 模型工厂
│   ├── model_strategies.py # 模型策略接口
│   ├── openai_strategy.py  # OpenAI模型策略
│   ├── qwen_strategy.py    # 阿里云通义千问模型策略
│   ├── zhihu_hot.py        # 知乎热榜获取
│   ├── zhihu_poster.py     # 知乎发布器
│   └── zhipu_strategy.py   # 智谱AI模型策略
├── frontend/               # 前端代码
│   └── app.py              # Streamlit应用
├── zhihu_answers/          # 生成的知乎回答
├── cookies/                # 知乎Cookie存储目录
├── .env                    # 环境变量文件
├── .env.example            # 环境变量示例文件
├── .gitignore              # Git忽略文件
├── README.md               # 项目说明文件
├── requirements.txt        # 依赖包列表
├── run.py                  # 运行脚本
└── test_model_factory.py   # 模型工厂测试脚本
```

## 贡献指南

欢迎提交Issue和Pull Request，一起完善这个项目！

## 许可证

MIT License

基于Playwright + LangGraph的知乎热榜回答Agent实现方案，包含UI界面和个性化回答功能。这个系统允许您上传自己的知识库，并根据您的风格自动生成知乎回答。

## 系统架构

```
frontend/ (Streamlit UI)
├── app.py                 # 主界面
backend/ (智能体核心)
├── knowledge_loader.py    # 知识库加载
├── agent_builder.py       # LangGraph智能体构建
├── zhihu_poster.py        # Playwright自动发布
├── vector_store/          # 用户知识库存储
```

## 功能特点

1. **个性化知识库**：
   - 支持上传TXT、PDF、MD格式的个人文档
   - 自动构建向量知识库，用于生成个性化回答

2. **智能回答生成**：
   - 基于LangGraph工作流的智能体
   - 支持多种回答风格和长度定制
   - 结合用户知识库和大模型能力

3. **自动发布**：
   - 使用Playwright自动登录知乎
   - 自动搜索问题并发布回答
   - 支持回答编辑和历史记录

## 安装与使用

### 环境要求

- Python 3.8+
- 以下任一大模型API密钥：
  - 智谱AI (GLM-4)
  - DeepSeek Chat
  - 阿里云通义千问
  - Moonshot AI (Kimi)
  - OpenAI

### 快速开始

1. **克隆仓库**：
   ```bash
   git clone https://github.com/yourusername/zhihu-ai-assistant.git
   cd zhihu-ai-assistant
   ```

2. **运行启动脚本**：
   ```bash
   python run.py
   ```
   启动脚本会自动检查依赖并安装缺失的包。

3. **配置环境变量**：
   在项目根目录创建`.env`文件，填入以下内容（只需要选择一个模型API密钥即可）：
   ```
   # 选择以下任一API密钥配置
   ZHIPU_API_KEY=your-zhipu-api-key        # 智谱AI
   DEEPSEEK_API_KEY=your-deepseek-api-key  # DeepSeek
   DASHSCOPE_API_KEY=your-dashscope-api-key # 阿里云通义千问
   KIMI_API_KEY=your-kimi-api-key          # Moonshot AI (Kimi)
   OPENAI_API_KEY=sk-your-openai-key       # OpenAI
   
   # 知乎相关配置
   ZHIHU_COOKIE_PATH=cookies/zhihu_cookies.json
   ```

### 使用流程

1. **知识库上传**：
   - 通过左侧面板上传TXT/PDF/MD格式的个人文档
   - 系统自动构建向量知识库

2. **回答问题**：
   - 选择知乎热榜问题或输入自定义问题
   - 设置回答风格和长度
   - 点击"生成回答"获取个性化回答

3. **内容发布**：
   - 预览生成的回答
   - 可以编辑回答内容
   - 点击"发布到知乎"自动提交回答

4. **历史记录**：
   - 可以保存生成的回答到历史记录
   - 随时查看和重用历史回答

## 首次使用

首次使用时，系统会自动打开浏览器并要求您登录知乎。登录后，系统会保存cookies以便后续使用。

## 扩展功能

1. **风格定制**：
   - 在UI中选择不同的回答风格
   - 支持专业严谨、幽默风趣、简洁明了等多种风格

2. **长度控制**：
   - 支持简短、中等、详细三种长度设置
   - 系统会根据设置自动控制回答字数

3. **安全性**：
   - 本地保存cookies，不会上传敏感信息
   - 支持手动编辑回答内容，确保内容安全

## 注意事项

- 请确保您的OpenAI API密钥有足够的额度
- 知乎账号可能会因为自动化操作受到限制，请谨慎使用
- 上传的知识文档仅保存在本地，不会上传到云端

## 许可证

MIT License