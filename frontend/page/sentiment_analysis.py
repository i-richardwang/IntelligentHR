import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from typing import List, Dict, Any
import asyncio

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend.ui_components import show_sidebar, show_footer, apply_common_styles
from backend.text_processing.classification.content_analysis_workflow import (
    TextContentAnalysisWorkflow,
)
from backend.text_processing.classification.content_analysis_core import (
    ContentAnalysisInput,
    ContentAnalysisResult,
)

import uuid

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# 初始化文本分类工作流
workflow = TextContentAnalysisWorkflow()


# 初始化会话状态
def initialize_session_state():
    """初始化 Streamlit 会话状态，设置默认值。"""
    default_states = {
        "classification_results": None,
        "df": None,
        "filtered_df": None,
        "context": "",
        "session_id": str(uuid.uuid4()),
        "is_processing": False,
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


def display_classification_result(result: ContentAnalysisResult):
    """将分析结果显示为表格"""
    df = pd.DataFrame(
        {
            "有效性": [result.validity],
            "情感倾向": [result.sentiment_class],
            "是否包含敏感信息": [result.sensitive_info],
        }
    )
    st.table(df)


async def batch_classify(texts: List[str], context: str, progress_bar, status_area):
    total_texts = len(texts)
    workflow = TextContentAnalysisWorkflow()
    results = []

    async def process_batch(batch):
        batch_results = await workflow.async_batch_analyze(
            batch, context, st.session_state.session_id, max_concurrency=3
        )
        results.extend(batch_results)
        return len(batch_results)

    batch_size = 3
    for i in range(0, total_texts, batch_size):
        batch = texts[i : i + batch_size]
        processed_count = await process_batch(batch)

        # 更新进度条和状态信息
        progress = (i + processed_count) / total_texts
        progress_bar.progress(progress)
        status_message = f"已处理: {i + processed_count}/{total_texts}"
        status_area.info(status_message)

        # 添加小延迟以允许UI更新
        await asyncio.sleep(0.05)

    return [result.model_dump() for result in results]


def display_info_message():
    """显示情感分析与标注工具的信息消息。"""
    st.info(
        """
    情感分析与标注功能使用大语言模型处理技术，帮助用户快速分析和分类大量文本数据。
    
    主要功能包括：
    - 文本有效性判断
    - 情感倾向分析
    - 是否敏感信息识别
    
    适用于各类需要快速理解和分类大量文本数据的场景，如客户反馈分析、社交媒体监控等。
    """
    )


def display_workflow():
    """显示情感分析与标注的工作流程。"""
    with st.expander("📋 查看情感分析与标注工作流程", expanded=False):

        with st.container(border=True):
            st.markdown(
                """
            1. **数据准备**: 
               - 输入单条文本或上传CSV文件
               - 指定文本的上下文或主题
               - 定义分类标签列表
            
            2. **文本分类**:
               - 系统自动判断文本有效性
               - 分类文本的情感倾向
               - 识别可能的敏感信息
            
            3. **结果展示**:
               - 显示每条文本的分类结果
               - 对于批量处理，以表格形式展示所有结果
            
            4. **结果导出**:
               - 提供分类结果的CSV下载选项
               - 便于进一步分析和报告生成
            
            5. **迭代优化**:
               - 根据分类结果调整上下文或标签
               - 重新运行分类以提高准确性
            """
            )


def main():
    st.title("🏷️ 情感分析与标注")
    st.markdown("---")

    display_info_message()
    display_workflow()

    st.markdown("## 情感分析与标注")
    with st.container(border=True):
        st.session_state.context = st.text_input(
            "请输入文本上下文或主题",
            value=st.session_state.context,
            placeholder="例如：员工调研",
        )

        tab1, tab2 = st.tabs(["直接输入", "上传CSV文件"])

        with tab1:
            with st.form("single_classification_form", border=False):
                text_to_classify = st.text_area("请输入要分析的文本", height=150)
                submit_button = st.form_submit_button("分析")

                if submit_button:
                    if text_to_classify and st.session_state.context:
                        st.session_state.session_id = str(
                            uuid.uuid4()
                        )  # 为单个分类任务生成新的session_id
                        with st.spinner("正在分析..."):
                            input_data = ContentAnalysisInput(
                                text=text_to_classify,
                                context=st.session_state.context,
                            )
                            result = workflow.analyze_text(
                                input_data, st.session_state.session_id
                            )
                        st.session_state.classification_results = result
                    else:
                        st.warning("请输入文本和上下文")

        with tab2:
            uploaded_file = st.file_uploader("上传CSV文件", type="csv")
            if uploaded_file is not None:
                try:
                    st.session_state.df = pd.read_csv(uploaded_file)
                    st.write("预览上传的数据：")
                    st.dataframe(st.session_state.df)

                    text_column = st.selectbox(
                        "选择包含要分析文本的列", st.session_state.df.columns
                    )

                    if st.button("开始批量分析"):
                        if st.session_state.context:
                            st.session_state.session_id = str(
                                uuid.uuid4()
                            )  # 为整个批量任务生成新的session_id
                            st.session_state.filtered_df = st.session_state.df[
                                [text_column]
                            ].copy()
                            st.session_state.current_batch_index = 0
                            st.session_state.total_rows = len(
                                st.session_state.filtered_df
                            )
                            st.session_state.progress = 0
                            st.session_state.is_processing = True
                        else:
                            st.warning("请输入上下文")

                except Exception as e:
                    st.error(f"处理CSV文件时出错：{str(e)}")

    if st.session_state.is_processing:
        st.markdown("## 批量分析进度")
        with st.container(border=True):
            total_rows = len(st.session_state.filtered_df)

            progress_bar = st.progress(0)
            status_area = st.empty()

            texts_to_classify = st.session_state.filtered_df[text_column].tolist()

            with st.spinner("正在批量分析..."):
                results = asyncio.run(
                    batch_classify(
                        texts_to_classify,
                        st.session_state.context,
                        progress_bar,
                        status_area,
                    )
                )

            for i, result in enumerate(results):
                st.session_state.filtered_df.loc[i, "有效性"] = result["validity"]
                st.session_state.filtered_df.loc[i, "情感倾向"] = result[
                    "sentiment_class"
                ]
                st.session_state.filtered_df.loc[i, "是否包含敏感信息"] = result[
                    "sensitive_info"
                ]

            st.success("批量分析完成！")
            st.session_state.classification_results = st.session_state.filtered_df
            st.session_state.is_processing = False

    # 显示分类结果
    if st.session_state.classification_results is not None:
        st.markdown("## 分析结果")
        with st.container(border=True):
            if isinstance(
                st.session_state.classification_results, ContentAnalysisResult
            ):
                # 单个文本分类结果
                display_classification_result(st.session_state.classification_results)
            elif isinstance(st.session_state.classification_results, pd.DataFrame):
                # 批量分类结果
                st.dataframe(st.session_state.classification_results)

                # 提供下载选项
                csv = st.session_state.classification_results.to_csv(
                    index=False
                ).encode("utf-8-sig")
                st.download_button(
                    label="下载分析结果CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )

    # 页脚
    show_footer()


main()
