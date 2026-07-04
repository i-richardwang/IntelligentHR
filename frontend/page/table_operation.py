import io
import os
import re
import sys
from typing import Dict, List, Tuple
import uuid
from unicodedata import category

import pandas as pd
import streamlit as st
from PIL import Image

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from backend.data_processing.table_operation.table_operation_workflow import (
    DataFrameWorkflow,
)
from frontend.ui_components import show_sidebar, show_footer, apply_common_styles

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# 初始化会话状态
if "workflow" not in st.session_state:
    st.session_state.workflow = DataFrameWorkflow()
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False
if "operation_result" not in st.session_state:
    st.session_state.operation_result = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "operation_steps" not in st.session_state:
    st.session_state.operation_steps = []
if "operation_result" not in st.session_state:
    st.session_state.operation_result = None


def main():
    """主函数，包含应用的主要逻辑和UI结构。"""
    st.title("🧮 智能数据整理")
    st.markdown("---")

    display_info_message()
    display_workflow()
    display_user_guide()

    handle_file_upload()
    if st.session_state.files_uploaded:
        process_user_query()

        if st.session_state.get("operation_result"):
            display_operation_result()
            display_feedback()

    show_footer()


def display_info_message():
    """
    显示智能数据整理的信息消息和隐私声明。
    """
    st.info(
        """
    本工具利用大模型的语义理解能力，通过自然语言交互实现多种表格操作，简化数据处理流程。

    系统能够理解并执行用户的自然语言指令，支持表格合并、数据重塑（宽转长、长转宽）和数据集比较等功能，还能够处理需要多个步骤才能完成的复杂数据处理需求。
    """
    )

    st.warning(
        "隐私声明：系统仅获取文件名称和字段名称，不读取也不存储上传的表格数据。表格预览仅在内存中临时加载，刷新页面后数据将被清除。"
    )


def display_workflow():
    """
    显示智能数据整理的工作流程。
    """
    with st.expander("📋 查看智能数据整理工作流程", expanded=False):
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                image = Image.open("frontend/assets/table_operation_workflow.png")
                st.image(image, caption="智能数据整理流程图", width="stretch")

            with col2:
                st.markdown(
                    """
                    1. **数据上传**
                        
                        支持CSV和Excel文件上传
                    
                    2. **自然语言指令输入**
                    
                        用户以对话方式输入数据处理需求，支持描述复杂的多步骤操作需求
        
                    3. **智能操作规划与执行**
                    
                        理解用户需求，自动规划所需操作步骤
                        
                        核心功能包括：
                          * 表格合并
                          * 数据重塑（宽转长、长转宽）
                          * 数据集比较
                        
                        支持多步骤复杂操作的顺序执行
        
                    4. **结果预览与导出**
                    
                        实时展示每个处理步骤的结果，支持导出每个处理步骤的结果
                """
                )


def display_user_guide():
    """
    使用选项卡展示用户指南，介绍支持的操作和如何描述需求。
    """
    with st.expander("📘 功能介绍与需求描述指南", expanded=False):
        st.markdown(
            """
        本工具支持多种数据处理操作。描述需求时，尽量指明操作的表格、期望的操作类型和关键信息，以提供AI处理的成功率。

        每个选项卡包含了特定操作类型的说明和示例。
        """
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["信息匹配", "数据转置", "数据对比", "垂直合并", "复杂分析"]
        )

        with tab1:
            st.markdown(
                """
            ### 信息匹配
            将两个或多个相关的表格中的信息进行匹配，类似于Excel中的VLOOKUP函数。

            **功能说明：**
            这个功能允许您将不同表格中的信息基于共同字段（如员工ID）进行匹配和合并，从而创建一个更全面的数据视图。

            **示例查询：**
            - "将员工薪资匹配到基本信息表"
            - "把培训记录表中的信息添加到员工主表中"
            """
            )

        with tab2:
            st.markdown(
                """
            ### 数据转置
            调整数据的结构，包括将宽表格转为长表格（多列转一列），或将长表格转为宽表格（一列转多列）

            **功能说明：**
            这个功能帮助您重新组织数据结构，使其更适合特定的分析需求或报告格式。

            **示例查询：**
            - "将员工月度考勤表从每月一列的格式转换为每个月份单独一行的格式"
            - "把多年绩效结果变成一列"
            """
            )

        with tab3:
            st.markdown(
                """
            ### 数据对比
            比较两个表格中指定信息的一致性或差异。

            **功能说明：**
            这个功能允许您对比两个表格中的特定字段，找出不一致或有差异的记录。

            **示例查询：**
            - "找出培训表中哪些员工不在在职花名册中"
            - "对比两个表中员工是否一致"
            """
            )

        with tab4:
            st.markdown(
                """
            ### 垂直合并
            将多个结构相似的表格垂直合并成一个大表。

            **功能说明：**
            这个功能帮助您将多个独立但结构相似的表格（如不同部门的报表或不同时期的数据）合并成一个综合表格。

            **示例查询：**
            - "垂直合并三个部门的员工名单"
            """
            )

        with tab5:
            st.markdown(
                """
            ### 复杂分析
            涉及多个步骤或多种操作类型的复杂数据处理需求。

            **功能说明：**
            这个功能允许您组合多个基本操作，执行更复杂的数据分析任务。系统将引导您逐步完成整个过程。

            **示例查询：**
            - "首先把考勤信息匹配到员工信息表，然后将结果按月份转置为每个员工一行的格式"
            - "先垂直合并各部门的员工信息表，然后再匹配上员工绩效"
            """
            )


def clean_filename(filename: str) -> str:
    # 移除文件扩展名
    name_without_extension = os.path.splitext(filename)[0]
    # 只保留中英文字符和数字
    cleaned_name = re.sub(r"[^\w\u4e00-\u9fff]+", "", name_without_extension)
    return cleaned_name if cleaned_name else "unnamed_file"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """清洗DataFrame的列名，移除特殊字符并替换空格。"""
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r"[^\w\u4e00-\u9fff]+", "_", regex=True)
    df.columns = df.columns.str.lower()
    return df


def handle_file_upload():
    """处理文件上传逻辑。"""
    st.markdown("## 数据上传")
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "选择CSV或Excel文件（可多选）",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                if file_extension == ".csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension in [".xlsx", ".xls"]:
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_names = xls.sheet_names
                    if len(sheet_names) > 1:
                        sheet_name = st.selectbox(
                            f"请选择要导入的sheet（{uploaded_file.name}）：",
                            sheet_names,
                        )
                    else:
                        sheet_name = sheet_names[0]
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                else:
                    st.error(f"不支持的文件格式：{file_extension}")
                    continue

                df = clean_column_names(df)

                df_name = clean_filename(uploaded_file.name)
                st.session_state.workflow.load_dataframe(df_name, df)

                # 提示用户字段过多可能导致执行成功率降低
                if df.shape[1] > 10:
                    st.warning(
                        f"注意：{uploaded_file.name} 包含超过10个字段，这可能导致执行成功率降低。"
                    )

            st.session_state.files_uploaded = True

        # 在文件上传的 container 中显示数据预览
        if st.session_state.files_uploaded:
            st.markdown("---")
            st.markdown("#### 上传的数据集预览")
            display_loaded_dataframes()


def display_loaded_dataframes():
    """使用标签页显示已加载的原始数据集预览。"""
    original_dataframes = st.session_state.workflow.get_original_dataframe_info()

    if not original_dataframes:
        st.info("还没有上传任何数据集。请先上传数据文件。")
        return

    # 创建标签页
    tabs = st.tabs(list(original_dataframes.keys()))

    # 为每个原始数据集创建一个标签页
    for tab, (name, info) in zip(tabs, original_dataframes.items()):
        with tab:
            df = st.session_state.workflow.get_dataframe(name)

            # 显示数据预览
            st.dataframe(df.head(5), width="stretch")

            # 显示简要信息
            st.caption(f"行数: {info['shape'][0]}, 列数: {info['shape'][1]}")
            st.caption(
                "注意：为了保证数据处理的准确性，若列名中含有特殊字符或空格，系统将自动进行清洗。"
            )


def process_user_query():
    """处理用户查询并显示结果。"""
    st.markdown("## 数据集操作")

    chat_container = st.container(border=True)
    input_placeholder = st.empty()

    display_conversation_history(chat_container)
    user_query = input_placeholder.chat_input("请输入您的数据集操作需求:")

    if user_query:
        display_user_input(chat_container, user_query)
        process_and_display_response(chat_container, user_query)


def display_conversation_history(container):
    """显示对话历史。"""
    with container:
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def display_user_input(container, user_query):
    """显示用户输入并保存到对话历史。"""
    with container:
        with st.chat_message("user"):
            st.markdown(user_query)
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_query}
    )


def process_and_display_response(container, user_query):
    """处理用户查询并显示响应。"""
    thinking_placeholder = st.empty()

    with thinking_placeholder:
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = st.session_state.workflow.process_query(user_query)

    thinking_placeholder.empty()

    display_assistant_response(container, result)

    # 保存 trace_id 到 session_state
    if "trace_id" in result:
        st.session_state.current_trace_id = result["trace_id"]

    # 新增：只有在执行操作时才设置 operation_result
    if result["next_step"] == "execute_operation":
        st.session_state.operation_result = result
    else:
        st.session_state.operation_result = None


def display_feedback():
    """显示简约的反馈元素并处理用户反馈，确保用户只能评价一次。"""
    if "current_trace_id" in st.session_state:
        st.markdown("---")
        st.markdown("##### 这次操作是否满足了您的需求？")

        # 初始化反馈状态
        if "feedback_given" not in st.session_state:
            st.session_state.feedback_given = False

        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            yes_button = st.button(
                "👍 是",
                key="feedback_yes",
                width="stretch",
                disabled=st.session_state.feedback_given,
            )
            if yes_button and not st.session_state.feedback_given:
                st.session_state.workflow.record_feedback(
                    trace_id=st.session_state.current_trace_id, is_useful=True
                )
                st.session_state.feedback_given = True

        with col2:
            no_button = st.button(
                "👎 否",
                key="feedback_no",
                width="stretch",
                disabled=st.session_state.feedback_given,
            )
            if no_button and not st.session_state.feedback_given:
                st.session_state.workflow.record_feedback(
                    trace_id=st.session_state.current_trace_id, is_useful=False
                )
                st.session_state.feedback_given = True

        with col3:
            if st.session_state.feedback_given:
                st.success("感谢您的反馈！")

        if st.session_state.feedback_given:
            del st.session_state.current_trace_id


def display_assistant_response(container, result):
    """显示助手的响应并保存到对话历史。"""
    with container:
        with st.chat_message("assistant"):
            if result["next_step"] == "need_more_info":
                message = result.get("message", "需要更多信息来处理您的请求。")
                st.markdown(message)
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": message}
                )
            elif result["next_step"] == "execute_operation":
                message = "操作执行成功！以下是执行的步骤：\n"
                st.markdown(message)
                st.session_state.operation_steps = result.get("operation", [])
                for i, step in enumerate(st.session_state.operation_steps, 1):
                    st.markdown(f"步骤 {i}: {step['tool_name']}")
                full_message = (
                    message
                    + "\n"
                    + "\n".join(
                        [
                            f"步骤 {i}: {step['tool_name']}"
                            for i, step in enumerate(
                                st.session_state.operation_steps, 1
                            )
                        ]
                    )
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": full_message}
                )
                st.session_state.operation_result = result
            elif result["next_step"] == "out_of_scope":
                message = result.get("message", "抱歉，您的请求超出了我的处理范围。")
                st.markdown(message)
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": message}
                )


def display_operation_result():
    """显示操作结果。"""
    if st.session_state.operation_result:
        result = st.session_state.operation_result

        st.markdown("---")
        st.markdown("## 操作结果")

        with st.container(border=True):
            for i, step in enumerate(st.session_state.operation_steps, 1):
                output_df_names = step["output_df_names"]
                for df_name in output_df_names:
                    if df_name in st.session_state.workflow.dataframes:
                        df = st.session_state.workflow.dataframes[df_name]
                        st.markdown(f"#### {df_name}")
                        st.caption(f"*由步骤 {i}: {step['tool_name']} 生成*")
                        st.dataframe(df)
                        provide_csv_download(df, df_name)
                st.markdown("---")


def provide_csv_download(df: pd.DataFrame, df_name: str):
    """为单个DataFrame提供CSV格式下载选项。"""
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"下载 {df_name} (CSV)",
        data=csv,
        file_name=f"{df_name}.csv",
        mime="text/csv",
    )


main()
