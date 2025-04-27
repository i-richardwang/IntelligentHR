import streamlit as st
import os
import sys
import pandas as pd
import uuid
from typing import List, Dict, Any
import asyncio

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend_demo.ui_components import show_sidebar, show_footer, apply_common_styles
from backend_demo.resume_management.storage.resume_storage_handler import (
    save_pdf_to_minio,
    calculate_file_hash,
    extract_text_from_url,
    calculate_url_hash,
    extract_text_from_pdf,
)
from backend_demo.resume_management.storage.resume_db_operations import (
    store_resume_record,
    get_resume_by_hash,
    update_resume_version,
    get_minio_path_by_id,
)
from backend_demo.resume_management.storage.resume_vector_storage import (
    store_raw_resume_text_in_milvus,
    search_similar_resumes,
    delete_resume_from_milvus,
)
from backend_demo.resume_management.storage.resume_comparison import compare_resumes
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# 初始化 session_state
if "step" not in st.session_state:
    st.session_state.step = "upload"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processing_results" not in st.session_state:
    st.session_state.processing_results = []
if "similar_resumes" not in st.session_state:
    st.session_state.similar_resumes = {}
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}


def main():
    st.title("📤 简历上传系统")
    st.markdown("---")

    display_info_message()
    display_workflow()

    if st.session_state.step == "upload":
        handle_upload()
    elif st.session_state.step == "process_and_review":
        process_and_review_uploads()

    show_footer()


def display_info_message():
    st.info(
        """
        智能简历上传系统支持PDF文件上传和URL输入两种方式。
        系统会自动提取简历内容，进行去重和存储。
        上传的简历将用于后续的智能匹配和分析。
        """
    )


def display_workflow():
    with st.expander("📄 查看简历上传工作流程", expanded=False):
        st.markdown(
            """
            1. **文件上传/URL输入**: 选择上传PDF文件或输入简历URL。
            2. **内容提取与去重检查**: 系统自动提取简历内容并检查是否存在重复。
            3. **相似度分析与智能比较**: 分析简历内容相似度，使用AI进行智能比较。
            4. **数据存储**: 根据分析结果自动存储简历信息。
            5. **处理结果展示**: 向用户显示最终的上传/处理结果。
            """
        )


def handle_upload():
    st.header("文件上传")
    with st.container(border=True):
        tab1, tab2 = st.tabs(["PDF上传", "URL输入"])

        with tab1:
            uploaded_files = st.file_uploader(
                "上传PDF简历", type=["pdf"], accept_multiple_files=True
            )
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                if st.button("开始处理上传的PDF文件"):
                    st.session_state.step = "process_and_review"

        with tab2:
            url = st.text_input("输入简历URL")
            if url and st.button("提交URL"):
                st.session_state.uploaded_files = [{"type": "url", "content": url}]
                st.session_state.step = "process_and_review"


def process_and_review_uploads():
    st.header("处理和审核简历")

    if not st.session_state.processing_results:
        with st.spinner("正在智能分析上传的简历..."):
            progress_bar = st.progress(0)
            for i, file in enumerate(st.session_state.uploaded_files):
                result = process_file(file)
                st.session_state.processing_results.append(result)
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_files))

    st.write("处理结果：")
    need_review = False
    for result in st.session_state.processing_results:
        st.write(
            f"文件名: {result['file_name']}, 状态: {result['status']}, 信息: {result['message']}"
        )
        if result["status"] == "潜在重复":
            need_review = True

    if need_review:
        st.subheader("智能比较相似简历")
        asyncio.run(compare_similar_resumes())

    confirm_uploads()


async def compare_similar_resumes():
    total_resumes = len(
        [r for r in st.session_state.processing_results if r["status"] == "潜在重复"]
    )
    progress_text = f"已比较 0/{total_resumes} 份简历"
    progress_bar = st.progress(0.0)
    progress_display = st.empty()
    progress_display.text(progress_text)

    resume_tabs = st.tabs(
        [
            f"简历 {i+1}"
            for i, r in enumerate(st.session_state.processing_results)
            if r["status"] == "潜在重复"
        ]
    )

    for i, (tab, result) in enumerate(
        zip(
            resume_tabs,
            [
                r
                for r in st.session_state.processing_results
                if r["status"] == "潜在重复"
            ],
        )
    ):
        with tab:
            st.subheader(f"文件名: {result['file_name']}")
            st.markdown(f"**状态**: {result['status']}")
            st.markdown(f"**消息**: {result['message']}")

            similar_resumes = st.session_state.similar_resumes.get(
                result["resume_hash"], []
            )

            with st.expander("查看相似简历详情", expanded=True):
                if similar_resumes:
                    df = pd.DataFrame(similar_resumes)
                    df["minio_path"] = df["resume_id"].apply(get_minio_path_by_id)
                    df["查看"] = df["minio_path"].apply(
                        lambda path: generate_minio_download_link(path) if path else "#"
                    )
                    st.dataframe(
                        df[["file_name", "upload_date", "similarity", "查看"]],
                        column_config={
                            "file_name": "文件名",
                            "upload_date": "上传日期",
                            "similarity": "相似度",
                            "查看": st.column_config.LinkColumn("简历链接"),
                        },
                        hide_index=True,
                    )
                else:
                    st.write("没有找到相似的简历。")

            # 使用AI进行比较
            with st.spinner("正在进行AI比较..."):
                session_id = str(uuid.uuid4())
                uploaded_resume_content = result["raw_content"]
                existing_resume_content = (
                    similar_resumes[0]["raw_content"] if similar_resumes else ""
                )

                comparison_result = await compare_resumes(
                    uploaded_resume_content, existing_resume_content, session_id
                )

                comparison_result = comparison_result.get(
                    "properties", comparison_result
                )

                st.session_state.comparison_results[result["resume_hash"]] = (
                    comparison_result
                )

            with st.expander("查看AI比较结果", expanded=False):
                st.json(comparison_result)

            progress_bar.progress((i + 1) / total_resumes)
            progress_display.text(f"已比较 {i+1}/{total_resumes} 份简历")


def confirm_uploads():
    st.header("处理结果")
    st.write("根据智能分析结果,系统自动处理如下：")

    for result in st.session_state.processing_results:
        if result["status"] == "潜在重复":
            comparison_result = st.session_state.comparison_results.get(
                result["resume_hash"]
            )
            if comparison_result:
                if comparison_result["is_same_candidate"]:
                    if comparison_result["latest_version"] == "uploaded_resume":
                        st.success(
                            f"{result['file_name']} 是同一候选人的最新版本简历，已更新到人才库。"
                        )
                        handle_latest_version(result)
                    else:
                        st.info(
                            f"{result['file_name']} 是同一候选人的旧版本简历，已保存为历史记录。"
                        )
                        handle_old_version(result)
                else:
                    st.success(
                        f"{result['file_name']} 已作为新候选人的简历保存到人才库中。"
                    )
                    handle_different_candidate(result)
            else:
                st.error(f"未能完成 {result['file_name']} 的智能比较,请稍后重试")
        elif result["status"] == "成功":
            st.success(f"{result['file_name']} 已成功添加到人才库中。")
        elif result["status"] == "已存在":
            st.info(f"{result['file_name']} 已存在于人才库中,无需重复添加。")
        else:
            st.error(f"{result['file_name']} 处理失败：{result['message']}")


def handle_latest_version(result):
    # 存储新简历到MySQL
    store_resume_record(
        result["resume_hash"],
        "pdf" if result.get("minio_path") else "url",
        result.get("file_name"),
        result.get("file_name") if not result.get("minio_path") else None,
        result.get("minio_path"),
        result["raw_content"],
    )

    # 删除Milvus中的旧版本并添加新版本
    similar_resumes = st.session_state.similar_resumes.get(result["resume_hash"], [])
    if similar_resumes:
        old_resume_id = similar_resumes[0]["resume_id"]
        delete_resume_from_milvus(old_resume_id)
        update_resume_version(old_resume_id, result["resume_hash"])

    # 存储新版本到Milvus
    store_raw_resume_text_in_milvus(
        result["resume_hash"], result["raw_content"], result["file_name"]
    )


def handle_old_version(result):
    # 仅存储到MySQL，标记为过时
    store_resume_record(
        result["resume_hash"],
        "pdf" if result.get("minio_path") else "url",
        result.get("file_name"),
        result.get("file_name") if not result.get("minio_path") else None,
        result.get("minio_path"),
        result["raw_content"],
        is_outdated=True,
        latest_resume_id=st.session_state.similar_resumes[result["resume_hash"]][0][
            "resume_id"
        ],
    )


def handle_different_candidate(result):
    # 不是同一候选人，存储到MySQL和Milvus
    store_resume_record(
        result["resume_hash"],
        "pdf" if result.get("minio_path") else "url",
        result.get("file_name"),
        result.get("file_name") if not result.get("minio_path") else None,
        result.get("minio_path"),
        result["raw_content"],
    )
    store_raw_resume_text_in_milvus(
        result["resume_hash"], result["raw_content"], result["file_name"]
    )


def process_file(file):
    if isinstance(file, dict) and file["type"] == "url":
        return process_url(file["content"])
    else:
        return process_pdf_file(file)


def process_pdf_file(file):
    file_hash = calculate_file_hash(file)
    existing_resume = get_resume_by_hash(file_hash)

    if existing_resume:
        return {
            "file_name": file.name,
            "status": "已存在",
            "message": "文件已存在于数据库中",
            "resume_hash": file_hash,
        }

    try:
        minio_path = save_pdf_to_minio(file)
        raw_content = extract_text_from_pdf(file)

        similar_resumes = search_similar_resumes(raw_content, top_k=1, threshold=0.9)

        if similar_resumes:
            st.session_state.similar_resumes[file_hash] = similar_resumes
            return {
                "file_name": file.name,
                "status": "潜在重复",
                "message": f"发现 {len(similar_resumes)} 份相似简历",
                "resume_hash": file_hash,
                "raw_content": raw_content,
                "minio_path": minio_path,
            }
        else:
            store_resume_record(
                file_hash, "pdf", file.name, None, minio_path, raw_content
            )
            store_raw_resume_text_in_milvus(file_hash, raw_content, file.name)
            return {
                "file_name": file.name,
                "status": "成功",
                "message": "上传成功并保存到数据库",
                "resume_hash": file_hash,
            }
    except Exception as e:
        return {
            "file_name": file.name,
            "status": "失败",
            "message": f"处理出错: {str(e)}",
            "resume_hash": file_hash,
        }


def process_url(url):
    try:
        content = asyncio.run(extract_text_from_url(url))
        url_hash = calculate_url_hash(content)

        existing_resume = get_resume_by_hash(url_hash)

        if existing_resume:
            return {
                "file_name": url,
                "status": "已存在",
                "message": "URL内容已存在于数据库中",
                "resume_hash": url_hash,
            }

        similar_resumes = search_similar_resumes(content, top_k=1, threshold=0.9)

        if similar_resumes:
            st.session_state.similar_resumes[url_hash] = similar_resumes
            return {
                "file_name": url,
                "status": "潜在重复",
                "message": f"发现 {len(similar_resumes)} 份相似简历",
                "resume_hash": url_hash,
                "raw_content": content,
            }
        else:
            store_resume_record(url_hash, "url", None, url, None, content)
            store_raw_resume_text_in_milvus(url_hash, content, url)
            return {
                "file_name": url,
                "status": "成功",
                "message": "URL简历已成功保存到数据库",
                "resume_hash": url_hash,
            }
    except Exception as e:
        return {
            "file_name": url,
            "status": "失败",
            "message": f"处理出错: {str(e)}",
            "resume_hash": None,
        }


def generate_minio_download_link(minio_path: str) -> str:
    if not minio_path:
        return "#"
    minio_base_url = os.getenv("MINIO_BASE_URL", "http://localhost:9000")
    bucket_name = os.getenv("MINIO_BUCKET_NAME", "resumes")
    return f"{minio_base_url}/{bucket_name}/{minio_path}"


def reset_state():
    st.session_state.step = "upload"
    st.session_state.uploaded_files = []
    st.session_state.processing_results = []
    st.session_state.similar_resumes = {}
    st.session_state.comparison_results = {}


main()
