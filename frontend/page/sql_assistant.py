# sql_assistant.py

import os
import sys
import uuid

import streamlit as st

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from langgraph.checkpoint.memory import MemorySaver

from frontend.ui_components import show_sidebar, show_footer, apply_common_styles
from backend.sql_assistant.graph.assistant_graph import run_sql_assistant

# 设置页面角色
st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        # 会话ID作为图执行的 thread_id，配合 checkpointer 支持多轮对话
        st.session_state.session_id = str(uuid.uuid4())
    if "checkpoint_saver" not in st.session_state:
        # 在整个会话内保持同一个状态保存器，保证多轮上下文可以延续
        st.session_state.checkpoint_saver = MemorySaver()


def display_info_message():
    """显示页面头部与使用说明"""
    st.title("💬 SQL 查询助手")
    st.markdown("---")

    st.info(
        """
    本工具利用大模型的语义理解能力，通过自然语言交互完成数据查询，简化取数流程。

    使用说明：
    1. 直接用自然语言描述您的数据查询需求（无需手动选表，系统会自动匹配相关数据表）
    2. 系统会先理解并改写您的需求，再生成并执行相应的 SQL 查询
    3. 支持多轮对话，可在上一次结果的基础上继续追问
    """
    )


def display_conversation_history():
    """使用聊天气泡显示对话历史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def run_query(user_query: str) -> str:
    """调用 SQL 助手图处理用户查询，返回面向用户的回复文本

    Args:
        user_query: 用户的自然语言查询

    Returns:
        str: 助手回复文本（含结果说明与数据表格，或错误/空结果提示）
    """
    try:
        result = run_sql_assistant(
            query=user_query,
            thread_id=st.session_state.session_id,
            checkpoint_saver=st.session_state.checkpoint_saver,
            user_id=None,
        )
    except Exception as e:
        return (
            f"抱歉，系统处理您的请求时出现错误：{str(e)}，"
            f"请稍后重试或联系数据团队寻求帮助。"
        )

    # run_sql_assistant 在执行出错时返回 {"error": ...}，正常时返回完整状态字典
    if not isinstance(result, dict):
        return "抱歉，未获取到有效的助手回复，请稍后重试。"
    if result.get("error"):
        return f"抱歉，处理您的请求时出现问题：{result['error']}"

    messages = result.get("messages", [])
    if not messages:
        return "抱歉，未获取到助手回复，请稍后重试。"

    last_message = messages[-1]
    return getattr(last_message, "content", str(last_message))


def process_user_query(user_query: str):
    """处理用户查询：显示用户消息、调用后端、渲染并记录助手回复"""
    # 显示并记录用户消息
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 生成并显示助手回复
    with st.chat_message("assistant"):
        with st.spinner("正在理解您的问题并查询数据，请稍候..."):
            answer = run_query(user_query)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def main():
    """主函数"""
    init_session_state()

    display_info_message()

    st.markdown("## 查询对话")
    # 展示历史对话
    display_conversation_history()

    # 用户输入
    user_query = st.chat_input("请输入您的数据查询问题，例如：统计各部门的在职人数")
    if user_query:
        process_user_query(user_query)

    # 显示页脚
    show_footer()


main()
