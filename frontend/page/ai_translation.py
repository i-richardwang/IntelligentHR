import asyncio
import os
import sys
import uuid
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
from asyncio import Semaphore

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend_demo.ui_components import show_sidebar, show_footer, apply_common_styles
from backend_demo.text_processing.translation.translator import Translator

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# 初始化翻译器
translator = Translator()

# 初始化 session state
if "translation_results" not in st.session_state:
    st.session_state.translation_results = None


def contains_chinese(text: str) -> bool:
    """
    检查文本是否包含超过50%的中文字符。

    Args:
        text (str): 要检查的文本。

    Returns:
        bool: 如果文本包含超过50%的中文字符，返回True；否则返回False。
    """
    if isinstance(text, str):
        chinese_chars = re.findall("[\u4e00-\u9fff]", text)
        return len(chinese_chars) / len(text) >= 0.5 if text else False
    return False


async def translate_text(text: str, text_topic: str) -> str:
    """
    异步翻译单个文本，如果文本主要是中文则直接返回原文。

    Args:
        text (str): 要翻译的文本。
        text_topic (str): 文本主题。

    Returns:
        str: 翻译后的文本、原中文文本或错误信息。
    """
    try:
        if contains_chinese(text):
            return text  # 如果文本主要是中文，直接返回原文
        session_id = str(uuid.uuid4())
        return await translator.translate(text, text_topic, session_id)
    except Exception as e:
        return f"翻译错误: {str(e)}"


async def batch_translate(
    texts: List[str], text_topic: str, session_id: str, max_concurrent: int = 3
) -> List[str]:
    """
    批量翻译文本，限制并发数量，并跳过主要是中文的文本。

    Args:
        texts (List[str]): 要翻译的文本列表。
        text_topic (str): 文本主题。
        session_id (str): 用于整个CSV文件的session ID。
        max_concurrent (int): 最大并发翻译数量，默认为3。

    Returns:
        List[str]: 翻译后的文本列表。
    """
    semaphore = Semaphore(max_concurrent)

    async def translate_with_semaphore(text: str) -> str:
        async with semaphore:
            if contains_chinese(text):
                return text  # 如果文本主要是中文，直接返回原文
            return await translator.translate(text, text_topic, session_id)

    tasks = [translate_with_semaphore(text) for text in texts]
    return await asyncio.gather(*tasks)


def display_translation_info() -> None:
    """显示翻译功能的介绍信息。"""
    st.info(
        """
    智能语境翻译是一个高效的多语言翻译工具，专为批量处理文本设计，通过上下文理解提高翻译准确性。
    本工具会自动检测并跳过主要由中文组成的文本，只翻译非中文文本。

    智能语境翻译适用于需要快速、准确翻译大量文本的各类场景，如多语言数据分析。
    """
    )


def upload_and_process_file() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    上传并处理CSV文件。

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: 包含上传的数据框和选中的文本列名。
    """
    uploaded_file = st.file_uploader("上传CSV文件", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("预览上传的数据：")
            st.dataframe(df)

            text_column = st.selectbox("选择包含要翻译文本的列", df.columns)
            return df, text_column
        except Exception as e:
            st.error(f"处理CSV文件时出错：{str(e)}")
    return None, None


def perform_translation(
    df: pd.DataFrame, text_column: str, text_topic: str, max_concurrent: int = 3
) -> pd.DataFrame:
    """
    执行批量翻译。

    Args:
        df (pd.DataFrame): 包含要翻译文本的数据框。
        text_column (str): 要翻译的文本列名。
        text_topic (str): 文本主题。
        max_concurrent (int): 最大并发翻译数量，默认为3。

    Returns:
        pd.DataFrame: 包含翻译结果的数据框。
    """
    texts_to_translate = df[text_column].tolist()
    session_id = str(uuid.uuid4())
    translated_texts = []

    async def translate_and_save(texts: List[str]) -> List[str]:
        results = await batch_translate(texts, text_topic, session_id, max_concurrent)
        translated_texts.extend(results)

        # 每翻译10个数据，保存一次临时结果
        if len(translated_texts) % 10 == 0 or len(translated_texts) == len(
            texts_to_translate
        ):
            temp_df = df.copy()
            temp_df["translated_text"] = translated_texts + [""] * (
                len(df) - len(translated_texts)
            )
            save_temp_results(temp_df, session_id)

        return results

    with st.spinner("正在批量翻译..."):
        asyncio.run(translate_and_save(texts_to_translate))

    df["translated_text"] = translated_texts
    return df


def save_temp_results(df: pd.DataFrame, session_id: str) -> None:
    """
    保存临时翻译结果到CSV文件。

    Args:
        df (pd.DataFrame): 包含翻译结果的数据框。
        session_id (str): 会话ID，用于生成唯一的文件名。
    """
    temp_dir = os.path.join("data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"translation_results_{session_id}.csv")
    df.to_csv(temp_file_path, index=False, encoding="utf-8-sig")


def display_translation_results(translation_results: pd.DataFrame) -> None:
    """
    显示翻译结果。

    Args:
        translation_results (pd.DataFrame): 包含翻译结果的数据框。
    """
    st.markdown("## 翻译结果")
    with st.container(border=True):
        if isinstance(translation_results, dict):
            with st.expander("查看翻译结果", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("原文")
                    st.markdown(translation_results["original"])
                with col2:
                    st.subheader("译文")
                    st.markdown(translation_results["translated"])
        elif isinstance(translation_results, pd.DataFrame):
            st.dataframe(translation_results)
            csv = translation_results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="下载翻译结果CSV",
                data=csv,
                file_name="translated_results.csv",
                mime="text/csv",
            )


def main() -> None:
    """主函数，包含智能语境翻译的整个流程。"""
    st.title("🌐 智能语境翻译")
    st.markdown("---")

    # 显示功能介绍
    display_translation_info()

    st.markdown("## 文本翻译")

    with st.container(border=True):
        text_topic = st.text_input(
            "请输入文本主题", placeholder="例如：员工反馈、绩效评价、工作报告等"
        )

        tab1, tab2 = st.tabs(["直接输入", "上传CSV文件"])

        with tab1:
            with st.form("single_translation_form", border=False):
                text_to_translate = st.text_area("请输入要翻译的文本", height=150)
                submit_button = st.form_submit_button("翻译")

                if submit_button and text_to_translate and text_topic:
                    if contains_chinese(text_to_translate):
                        st.info("检测到输入的文本主要是中文，无需翻译。")
                        st.session_state.translation_results = {
                            "original": text_to_translate,
                            "translated": text_to_translate,
                        }
                    else:
                        with st.spinner("正在翻译..."):
                            translated_text = asyncio.run(
                                translate_text(text_to_translate, text_topic)
                            )
                        st.session_state.translation_results = {
                            "original": text_to_translate,
                            "translated": translated_text,
                        }

        with tab2:
            df, text_column = upload_and_process_file()
            if df is not None and st.button("开始批量翻译") and text_topic:
                st.session_state.translation_results = perform_translation(
                    df, text_column, text_topic
                )

    if st.session_state.translation_results is not None:
        display_translation_results(st.session_state.translation_results)

    # 页脚
    show_footer()


main()
