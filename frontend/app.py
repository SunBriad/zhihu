import streamlit as st
import sys
import os
import time

# 添加后端目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.knowledge_loader import load_knowledge_base
from backend.agent_builder import get_agent_executor
from backend.config import update_model_config
from backend.zhihu_poster import post_to_zhihu
from backend.zhihu_hot import get_zhihu_hot_questions

st.title("知乎热榜AI助手")

# 知识库上传区
with st.sidebar:
    st.header("个性化设置")
    
    # API设置（可折叠）
    with st.expander("API设置"):
        # 从后端导入支持的模型提供商
        from backend.config import SUPPORTED_PROVIDERS
        
        # 模型提供商选择
        provider_options = list(SUPPORTED_PROVIDERS.keys())
        provider_display_names = [f"{SUPPORTED_PROVIDERS[p]['name']} ({p})" for p in provider_options]
        
        provider_index = st.selectbox(
            "模型提供商",
            range(len(provider_options)),
            format_func=lambda i: provider_display_names[i],
            help="选择使用的模型提供商"
        )
        
        provider = provider_options[provider_index]
        
        # 显示API密钥输入框
        api_key = st.text_input(
            f"{SUPPORTED_PROVIDERS[provider]['name']} API Key", 
            value=os.environ.get(SUPPORTED_PROVIDERS[provider].get('env_var', '') or "OPENAI_API_KEY", ""), 
            type="password",
            help=f"输入您的{SUPPORTED_PROVIDERS[provider]['name']}API密钥"
        )
        
        # 模型选择
        analysis_model = st.selectbox(
            "分析模型",
            ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o","deepseek-chat","deepseek-reasoner"],
            index=0,
            help="用于问题分析的模型（仅OpenAI模式下有效）"
        )
        
        generation_model = st.selectbox(
            "生成模型",
            ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o","deepseek-chat","deepseek-reasoner"],
            index=1,
            help="用于回答生成的模型（仅OpenAI模式下有效）"
        )
        
        if st.button("保存API设置"):
            # 获取当前选择的提供商的环境变量名
            env_var = SUPPORTED_PROVIDERS[provider].get('env_var')
            
            # 更新环境变量
            if env_var:
                os.environ[env_var] = api_key
            
            # 更新模型配置
            update_model_config(
                api_key=api_key,
                analysis_model=analysis_model,
                generation_model=generation_model,
                provider=provider
            )
            st.success(f"{SUPPORTED_PROVIDERS[provider]['name']} API设置已保存！")
    
    uploaded_files = st.file_uploader("上传知识文档", 
                    type=["txt", "md", "pdf"], 
                    accept_multiple_files=True)
    
    # 风格设置
    tone_options = ["专业严谨", "幽默风趣", "简洁明了", "深度思考"]
    selected_tone = st.selectbox("选择回答风格", tone_options)
    
    # 长度设置
    length_options = ["简短", "中等", "详细"]
    selected_length = st.selectbox("选择回答长度", length_options)
    
    if uploaded_files:
        with st.spinner("构建知识库中..."):
            load_knowledge_base(uploaded_files)
            st.success("知识库更新完成！")

# 热榜问题选择区
st.header("知乎热榜问题")

# 添加刷新按钮
col1, col2 = st.columns([4, 1])
with col1:
    st.write("从知乎获取实时热榜问题")
with col2:
    refresh = st.button("刷新热榜")

# 使用缓存机制，避免频繁请求知乎API
if 'hot_questions' not in st.session_state or refresh:
    with st.spinner("正在获取知乎热榜..."):
        try:
            # 获取知乎热榜问题
            hot_questions = get_zhihu_hot_questions(limit=10)
            st.session_state.hot_questions = hot_questions
            st.session_state.last_update_time = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error(f"获取知乎热榜失败: {str(e)}")
            # 使用备用问题
            if 'hot_questions' not in st.session_state:
                st.session_state.hot_questions = [
                    "AI会取代程序员吗？", 
                    "如何学习大模型？",
                    "ChatGPT对教育行业有什么影响？",
                    "2024年值得关注的科技趋势有哪些？",
                    "如何平衡工作与生活？"
                ]

# 显示最后更新时间
if 'last_update_time' in st.session_state:
    st.caption(f"最后更新时间: {st.session_state.last_update_time}")

selected_q = st.selectbox("选择热榜问题", st.session_state.hot_questions)

# 自定义问题输入
custom_q = st.text_input("或者输入自定义问题")
if custom_q:
    selected_q = custom_q

# 回答生成区
if st.button("生成回答"):
    with st.spinner("智能思考中..."):
        try:
            agent = get_agent_executor()
            response = agent.invoke({
                "question": selected_q,
                "tone": selected_tone,
                "length": selected_length
            })
            st.markdown("### 生成的回答:")
            st.markdown(response['answer'])
            
            # 保存回答用于后续发布
            st.session_state.zhihu_answer = response['answer']
        except Exception as e:
            st.error(f"生成回答时出错: {str(e)}")

# 回答发布区
if 'zhihu_answer' in st.session_state:
    st.header("发布到知乎")
    
    # 添加编辑功能
    edited_answer = st.text_area("编辑回答", st.session_state.zhihu_answer, height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("重新生成"):
            st.session_state.pop('zhihu_answer', None)
            st.rerun()
    
    with col2:
        if st.button("发布到知乎"):
            with st.spinner("发布中..."):
                try:
                    success = post_to_zhihu(
                        question=selected_q,
                        answer=edited_answer
                    )
                    if success:
                        st.success("回答已成功发布到知乎！")
                    else:
                        st.warning("自动发布失败，回答已保存到本地文件。请查看控制台输出获取更多信息。")
                except Exception as e:
                    st.error(f"发布失败: {str(e)}")
                    # 保存回答到文件，作为备份
                    try:
                        save_path = f"zhihu_answers/{selected_q[:20]}.md"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(f"# {selected_q}\n\n{edited_answer}")
                        st.info(f"回答已保存到文件: {save_path}")
                    except Exception as save_error:
                        st.error(f"保存回答到文件时出错: {str(save_error)}")

# 添加历史记录功能
st.sidebar.header("历史记录")
if 'history' not in st.session_state:
    st.session_state.history = []

if 'zhihu_answer' in st.session_state and st.button("保存到历史记录"):
    st.session_state.history.append({
        "question": selected_q,
        "answer": st.session_state.zhihu_answer,
        "tone": selected_tone,
        "length": selected_length,
        "time": st.session_state.get("current_time", "未知时间")
    })
    st.success("已保存到历史记录！")

# 显示历史记录
if st.session_state.history:
    with st.sidebar.expander("查看历史记录"):
        for i, item in enumerate(st.session_state.history):
            st.write(f"**问题 {i+1}**: {item['question']}")
            if st.button(f"查看回答 {i+1}"):
                st.session_state.zhihu_answer = item['answer']
                st.rerun()