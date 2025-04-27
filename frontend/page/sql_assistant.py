# sql_assistant.py

from typing import Dict, Any
import streamlit as st
import os
import sys
import uuid
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend_demo.ui_components import show_sidebar, show_footer, apply_common_styles
from backend_demo.sql_assistant.result_interpreter import create_result_interpreter
from backend_demo.sql_assistant.sql_executor import create_sql_executor
from backend_demo.sql_assistant.sql_generation import SQLGenerator
from backend_demo.sql_assistant.query_enhancement import QueryEnhancer
from backend_demo.sql_assistant.db_schema import create_schema_manager

# 设置页面角色
st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

def init_session_state():
    """初始化会话状态"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "selected_table" not in st.session_state:
        st.session_state.selected_table = None
    if "schema_manager" not in st.session_state:
        st.session_state.schema_manager = create_schema_manager()
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_trace_id" not in st.session_state:
        st.session_state.current_trace_id = None


def display_header():
    """显示页面头部"""
    st.title("💬 SQL查询助手")
    st.markdown("---")

    st.info(
        """
    本工具利用大模型的语义理解能力，通过自然语言交互实现SQL查询，简化数据查询流程。
    
    使用说明：
    1. 选择要查询的表
    2. 用自然语言描述您的问题
    3. 系统会先理解您的需求，然后生成相应的SQL查询
    """
    )


def select_table() -> str:
    """
    选择要查询的表
    
    Returns:
        str: 选中的表名
    """
    try:
        tables = st.session_state.schema_manager.get_all_tables()
        table_options = {
            f"{table['name']} ({table['comment']})": table["name"]
            for table in tables
        }

        selected = st.selectbox(
            "选择要查询的表:",
            options=list(table_options.keys()),
            key="table_selector"
        )

        if selected:
            table_name = table_options[selected]
            if table_name != st.session_state.selected_table:
                st.session_state.selected_table = table_name
                st.session_state.conversation_history = []  # 清空对话历史
            return table_name
    except Exception as e:
        st.error(f"获取表列表失败: {str(e)}")
    return None


def create_executors(table_name: str) -> Dict[str, Any]:
    """
    创建执行器实例
    
    Args:
        table_name: 表名
        
    Returns:
        Dict[str, Any]: 包含所有执行器的字典
    """
    return {
        "query_enhancer": QueryEnhancer(),
        "sql_generator": SQLGenerator(),
        "sql_executor": create_sql_executor(allowed_table=table_name),
        "result_interpreter": create_result_interpreter(),
    }


def display_conversation_history():
    """使用聊天气泡显示对话历史"""
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                # 用户消息简单显示
                st.markdown(message["content"])
            else:
                # 助手消息包含多个部分
                # 1. 首先显示自然语言解释
                st.markdown(message["content"])

                # 2. 如果有查询理解说明，显示在可展开区域
                if "query_understanding" in message:
                    with st.expander("💭 查询理解过程"):
                        st.markdown(message["query_understanding"])

                # 3. 显示SQL（使用可展开的形式）
                if "sql" in message:
                    with st.expander("🔍 查看生成的SQL"):
                        st.code(message["sql"], language="sql")

                # 4. 如果有结果数据，显示数据预览
                if "results" in message and message["results"]:
                    with st.expander("📊 查看数据详情"):
                        st.dataframe(
                            message["results"],
                            use_container_width=True,
                            height=min(len(message["results"]) * 35 + 38, 250)
                        )


def add_message(role: str, content: Dict[str, Any]):
    """
    添加消息到对话历史
    
    Args:
        role: 消息角色（user/assistant）
        content: 消息内容
    """
    message = {"role": role, **content}
    st.session_state.conversation_history.append(message)


def process_user_query(user_query: str, table_name: str, chat_container):
    """
    处理用户查询并实时显示进度
    
    Args:
        user_query: 用户的查询
        table_name: 表名
        chat_container: 聊天容器
    """
    # 显示用户输入
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_query)
        add_message("user", {"content": user_query})

        # 创建一个助手消息占位符
        with st.chat_message("assistant"):
            progress_placeholder = st.empty()

            try:
                # 获取表结构信息
                table_schema = st.session_state.schema_manager.format_schema_for_llm(
                    table_name)
                executors = create_executors(table_name)

                # 1. 增强查询
                progress_placeholder.markdown("🤔 正在理解您的问题...")
                enhancement_result = executors['query_enhancer'].enhance_query(
                    user_query=user_query,
                    table_schema=table_schema,
                    session_id=st.session_state.session_id
                )
                enhanced_query = enhancement_result['enhanced_query']
                
                # 2. 生成SQL
                progress_placeholder.markdown("📝 正在构建查询...")
                sql_result = executors['sql_generator'].generate_sql(
                    enhanced_query=enhanced_query,
                    table_schema=table_schema,
                    session_id=st.session_state.session_id
                )
                sql_query = sql_result['sql']
                st.session_state.current_trace_id = sql_result['trace_id']

                # 3. 执行SQL
                progress_placeholder.markdown("⚡ 正在执行查询...")
                execution_result = executors['sql_executor'].execute_sql(sql_query)

                if not execution_result.success:
                    progress_placeholder.markdown(
                        f"❌ 查询执行失败: {execution_result.error_message}")
                    add_message("assistant", {
                        "content": f"❌ 查询执行失败: {execution_result.error_message}",
                        "query_understanding": enhancement_result['reasoning'],
                        "sql": sql_query
                    })
                    return

                if not execution_result.data:
                    progress_placeholder.markdown("ℹ️ 未找到匹配的数据。")
                    add_message("assistant", {
                        "content": "未找到匹配的数据。",
                        "query_understanding": enhancement_result['reasoning'],
                        "sql": sql_query,
                        "results": []
                    })
                    return

                # 4. 解释结果
                progress_placeholder.markdown("🧐 正在分析结果...")
                interpretation_result = executors['result_interpreter'].interpret_results(
                    user_query=user_query,
                    sql_query=sql_query,
                    query_results=execution_result.data,
                    table_schema=table_schema,
                    session_id=st.session_state.session_id
                )

                # 5. 显示完整回复
                progress_placeholder.empty()  # 清除进度提示

                # 显示自然语言解释
                st.markdown(interpretation_result['answer'])

                # 显示查询理解过程
                with st.expander("💭 查询理解过程"):
                    st.markdown(enhancement_result['reasoning'])

                # 显示SQL
                with st.expander("🔍 查看生成的SQL"):
                    st.code(sql_query, language="sql")

                # 显示数据
                with st.expander("📊 查看数据详情"):
                    st.dataframe(
                        execution_result.data,
                        use_container_width=True,
                        height=min(len(execution_result.data) * 35 + 38, 250)
                    )

                # 显示执行统计
                st.caption(
                    f"✨ 查询执行时间: {execution_result.execution_time:.2f}秒, "
                    f"返回数据量: {execution_result.affected_rows}条"
                )

                # 添加到对话历史
                add_message("assistant", {
                    "content": interpretation_result['answer'],
                    "query_understanding": enhancement_result['reasoning'],
                    "sql": sql_query,
                    "results": execution_result.data
                })

            except Exception as e:
                error_message = f"❌ 处理查询时发生错误: {str(e)}"
                progress_placeholder.markdown(error_message)
                add_message("assistant", {"content": error_message})
                return


def main():
    """主函数"""
    init_session_state()
    display_header()

    # 显示侧边栏
    show_sidebar()

    # 表选择和表结构展示
    st.markdown("## 数据表选择")
    with st.container(border=True):
        table_name = select_table()
        if table_name:
            if st.button("📋 查看表结构", use_container_width=True):
                with st.expander("表结构信息", expanded=True):
                    st.code(
                        st.session_state.schema_manager.format_schema_for_llm(
                            table_name),
                        language="markdown",
                    )

    if not table_name:
        return

    st.markdown("## 查询对话")
    # 创建对话界面
    chat_container = st.container(border=True)
    with chat_container:
        display_conversation_history()

    # 用户输入
    user_query = st.chat_input(
        "请输入您的问题:",
        key="query_input",
    )

    if user_query:
        process_user_query(user_query, table_name, chat_container)

    # 显示页脚
    show_footer()


main()