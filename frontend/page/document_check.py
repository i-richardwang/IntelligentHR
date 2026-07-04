import os
import sys
import streamlit as st
import asyncio
from typing import List, Dict, Any
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend.ui_components import show_sidebar, show_footer, apply_common_styles
from backend.document_check.document_check_core import process_document
import unstructured_client
from unstructured_client.models import operations, shared


st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# Unstructured文档解析客户端初始化
client = unstructured_client.UnstructuredClient(
    api_key_auth="",
    server_url=os.getenv("UNSTRUCTURED_API_URL", "http://localhost:8000"),
)


def parse_and_filter_document(file):
    """解析上传的文档并过滤短元素"""
    filename = file.name
    content = file.read()

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=content,
                file_name=filename,
            ),
            languages=["chi_sim"],
            extract_image_block_types=["Image", "Table"],
            strategy="hi_res",
        ),
    )

    res = client.general.partition(request=req)

    # 过滤掉长度小于5的元素
    filtered_elements = [
        element for element in res.elements if len(element.get("text", "")) >= 5
    ]
    return filtered_elements


def display_check_results(results: List[Dict[str, Any]]):
    """显示文档检查结果"""
    st.subheader("文档检查结果")

    # 创建一个选项卡列表，每个页面一个选项卡
    tabs = st.tabs([f"第 {result['page_number']} 页" for result in results])

    for i, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            if result["corrections"]:
                # 创建一个数据框来显示所有修改
                df = pd.DataFrame(result["corrections"])
                df = df.rename(
                    columns={
                        "element_id": "元素ID",
                        "original_text": "原始文本",
                        "suggestion": "修改建议",
                        "correction_reason": "修改理由",
                    }
                )

                # 显示数据框
                st.dataframe(df, width="stretch")

                # 为每个修改创建一个详细视图
                for correction in result["corrections"]:
                    with st.expander(f"详细信息 - 元素ID: {correction['element_id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**原始文本:**")
                            st.text(correction["original_text"])
                        with col2:
                            st.markdown("**修改建议:**")
                            st.text(correction["suggestion"])
                        st.markdown("**修改理由:**")
                        st.write(correction["correction_reason"])
            else:
                st.info("此页面未发现需要修改的内容。")

    # 添加一个汇总信息
    total_corrections = sum(len(result["corrections"]) for result in results)
    st.sidebar.metric("总修改建议数", total_corrections)


def main():
    st.title("🔍 智能文档检查工具")
    st.markdown("---")

    st.info(
        """
        智能文档检查工具利用先进的自然语言处理技术，帮助您快速检查文档中的错别字和表述不通顺的问题。
        支持多种文档格式，包括PDF、Word、PowerPoint等。上传您的文档，让我们开始检查吧！
        """
    )

    st.markdown("## 文档上传")
    with st.container(border=True):
        # 初始化 session state
        if "uploaded_file" not in st.session_state:
            st.session_state.uploaded_file = None
        if "check_results" not in st.session_state:
            st.session_state.check_results = None

        uploaded_file = st.file_uploader("上传文档", type=["pdf", "docx", "pptx"])

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

        if st.button("开始检查") and st.session_state.uploaded_file:
            with st.spinner("正在解析和检查文档..."):
                # 解析文档
                document_content = parse_and_filter_document(
                    st.session_state.uploaded_file
                )
                # 检查文档
                st.session_state.check_results = asyncio.run(
                    process_document(document_content)
                )
            st.success("文档检查完成！")

    if st.session_state.check_results:
        st.markdown("## 文档检查结果")
        with st.container(border=True):
            display_check_results(st.session_state.check_results)

    # 显示页脚
    show_footer()


main()
