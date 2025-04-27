import streamlit as st
from PIL import Image
import pandas as pd
import sys
import os
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4
import asyncio
import aiohttp

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend_demo.ui_components import show_sidebar, show_footer, apply_common_styles
from backend_demo.data_processing.data_cleaning.data_processor import (
    initialize_vector_store,
    get_entity_retriever,
)
from backend_demo.data_processing.data_cleaning.verification_workflow import (
    EntityVerificationWorkflow,
    ProcessingStatus,
)

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# 初始化会话状态
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "batch_results_df" not in st.session_state:
    st.session_state.batch_results_df = None

# 定义实体类型选项
ENTITY_TYPES = {
    "公司名称": {
        "collection_name": "company_data",
        "original_field": "company_name",
        "standard_field": "standard_name",
        "validation_instructions": """
        请特别注意，有效的公司名称应该是具体的、可识别的企业实体。
        例如，"科技公司"这样的泛称应被视为无效，而"阿里巴巴"则是有效的。
        """,
        "analysis_instructions": """
        在分析搜索结果时，请特别注意识别公司的正式名称、简称、曾用名等。
        需要推断该公司是否是某家知名公司的合同主体变体或其子公司。
        如果是，最终提供的应该是这家更普遍和知名的公司名称。
        例如，如果搜索结果显示"淘宝（中国）软件有限公司"是阿里巴巴集团的子公司，
        那么应该将"阿里巴巴"识别为标准化的公司名称。
        """,
        "verification_instructions": """
        在验证过程中，请考虑公司的各种可能的名称形式，包括全称、简称、品牌名等。
        即使名称表述不同，只要指向同一个母公司或集团，应视为匹配。
        例如，"阿里巴巴"和"阿里巴巴集团控股有限公司"应该被视为匹配。
        但对于名称表述相近，但指向不同公司的，应视为不匹配。
        例如，"海航"和"海尔"是不同的公司，应该被视为不匹配。
        """,
    },
    "学校名称": {
        "collection_name": "school_data",
        "original_field": "school_name",
        "standard_field": "standard_name",
        "validation_instructions": """
        有效的学校名称应该是具体的教育机构名称。
        例如，"大学"这样的泛称应被视为无效，而"北京大学"则是有效的。
        """,
        "analysis_instructions": """
        分析时注意识别学校的官方名称、常用简称等。
        需要考虑学校可能的更名历史和分校情况。
        例如，如果搜索结果显示"清华大学深圳研究生院"，
        应该将"清华大学"识别为标准化的学校名称。
        """,
        "verification_instructions": """
        验证时考虑学校的不同称呼，包括全称、简称、俗称等。
        只要指向同一所学校，即使表述不同也应视为匹配。
        例如，"北大"和"北京大学"应该被视为匹配。
        """,
    },
}


def initialize_workflow(
    entity_type: str,
    skip_validation: bool,
    skip_search: bool,
    skip_retrieval: bool,
    search_tool: str,
) -> EntityVerificationWorkflow:
    """
    初始化实体验证工作流。

    Args:
        entity_type (str): 实体类型。
        skip_validation (bool): 是否跳过输入验证。
        skip_search (bool): 是否跳过网络搜索和分析。
        skip_retrieval (bool): 是否跳过向量检索和匹配。
        search_tool (str): 使用的搜索工具（"duckduckgo" 或 "tavily"）。

    Returns:
        EntityVerificationWorkflow: 初始化后的工作流对象。
    """
    entity_info = ENTITY_TYPES[entity_type]
    try:
        collection = initialize_vector_store(entity_info["collection_name"])
    except ValueError as e:
        st.error(f"初始化向量存储时出错：{str(e)}")
        st.error("请确保已通过数据库管理界面创建并导入数据到相应的集合。")
        return None

    retriever = get_entity_retriever(collection, entity_type)
    return EntityVerificationWorkflow(
        retriever=retriever,
        entity_type=entity_type,
        original_field="original_name",
        standard_field="standard_name",
        validation_instructions=entity_info["validation_instructions"],
        analysis_instructions=entity_info["analysis_instructions"],
        verification_instructions=entity_info["verification_instructions"],
        skip_validation=skip_validation,
        skip_search=skip_search,
        skip_retrieval=skip_retrieval,
        search_tool=search_tool,
    )


def main():
    st.title("🏢 自动化数据清洗")
    st.markdown("---")

    # 显示功能介绍
    display_info_message()

    # 显示工作流程
    display_workflow()

    # 任务设置部分
    st.markdown("## 任务设置")

    with st.container(border=True):
        st.markdown("##### 数据清洗设置")

        # 使用更友好的业务语言
        entity_type = st.radio(
            "选择要清洗的数据类型",
            ["公司名称", "学校名称"],
            help="选择您需要标准化的数据类型。",
        )

        # 工作流程选项
        st.markdown("##### 清洗流程设置")
        col1, col2, col3 = st.columns(3)
        with col1:
            skip_validation = st.checkbox(
                "跳过输入验证",
                value=False,
                help="如果确信输入数据有效，可以跳过验证步骤。",
            )
        with col2:
            skip_search = st.checkbox(
                "跳过网络搜索",
                value=False,
                help="对于知名实体，可以直接进行数据库匹配，跳过网络搜索步骤。",
            )
        with col3:
            skip_retrieval = st.checkbox(
                "跳过数据库匹配",
                value=False,
                help="如果处理全新数据，可以跳过与已有数据库的匹配步骤。",
            )

        # 添加搜索工具选择
        search_tool = st.selectbox(
            "选择搜索工具",
            ["duckduckgo", "tavily"],
            help="选择用于网络搜索的工具。",
        )

    # 初始化工作流
    workflow = initialize_workflow(
        entity_type, skip_validation, skip_search, skip_retrieval, search_tool
    )

    st.markdown("## 数据清洗")

    with st.container(border=True):
        tab1, tab2 = st.tabs(["单个样本测试", "批量数据清洗"])

        with tab1:
            asyncio.run(single_entity_verification(workflow, entity_type))

        with tab2:
            batch_processing(workflow, entity_type)

    # 页脚
    show_footer()


def display_info_message():
    """
    显示自动化数据清洗的功能介绍。
    """
    st.info(
        """
    自动化数据清洗工具集成了大语言模型的推理和工具调用能力，实现高效精准的数据标准化。

    系统通过多阶段验证流程，智能识别和验证输入的实体名称，并利用向量检索技术在数据库中进行快速匹配。
    适用于需要大规模标准化和验证各类实体名称的数据处理场景。
    """
    )


def display_workflow():
    """
    显示自动化数据清洗工具的工作流程。
    """
    with st.expander("🏢 查看自动化数据清洗工作流程", expanded=False):
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                image = Image.open("frontend_demo/assets/data_cleaning_workflow.png")
                st.image(image, caption="自动化数据清洗流程图", use_container_width=True)

            with col2:
                st.markdown(
                    """
                    <div class="workflow-container">
                        <div class="workflow-step">
                            <strong>1. 智能数据验证</strong>: 利用大语言模型的自然语言理解能力，智能识别和初步验证输入的实体名称。
                        </div>
                        <div class="workflow-step">
                            <strong>2. 多源网络搜索</strong>: 调用多个搜索引擎API，全面收集实体相关信息，为后续分析提供丰富数据支持。
                        </div>
                        <div class="workflow-step">
                            <strong>3. 大模型推理分析</strong>: 运用大语言模型的推理能力，从搜索结果中提取关键信息，如实体全称、简称等。
                        </div>
                        <div class="workflow-step">
                            <strong>4. 向量检索匹配</strong>: 将处理后的实体信息转化为向量，在预构建的大规模实体向量数据库中进行高效、精准的相似度匹配。
                        </div>
                        <div class="workflow-step">
                            <strong>5. 结果验证与输出</strong>: 大语言模型对多源信息和匹配结果进行综合分析和验证，生成最终的标准化实体名称。
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


async def single_entity_verification(
    workflow: EntityVerificationWorkflow, entity_type: str
):
    """
    处理单个实体名称的标准化。

    Args:
        workflow (EntityVerificationWorkflow): 实体名称标准化的工作流对象。
        entity_type (str): 实体类型。
    """
    with st.form(key="single_entity_form"):
        entity_name = st.text_input(
            f"输入{entity_type}",
            placeholder=f"例如：{'阿里巴巴' if entity_type == '公司名称' else '北京大学'}",
        )
        submit_button = st.form_submit_button("标准化")
        if submit_button and entity_name:
            with st.spinner("正在标准化..."):
                session_id = str(uuid4())
                result = await workflow.run(entity_name, session_id=session_id)
            display_single_result(result, entity_type)


def display_single_result(result: Dict[str, Any], entity_type: str):
    st.success("数据处理完成！")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"最终{entity_type}", result["final_entity_name"])
    with col2:
        st.metric("标准化状态", result["status"].value)

    with st.expander("查看详细信息"):
        # 显示基本信息
        st.subheader("基本信息")
        display_info = {
            "原始输入": result["original_input"],
            "最终实体名称": result["final_entity_name"],
            "标准化状态": result["status"].value,
            "是否有效输入": result["is_valid"],
            "识别的实体名称": result.get("identified_entity_name"),
            "检索的实体名称": result.get("retrieved_entity_name"),
        }
        st.json(display_info)

        # 添加专门的搜索结果展示区域
        st.subheader("网络搜索结果")
        if result.get("search_results"):
            st.text_area(
                label="搜索结果详情",
                value=result["search_results"],
                height=200,
                disabled=True,
                key="search_results",
            )
        elif result.get("search_results") is None:
            st.info("未进行网络搜索")
        else:
            st.info("网络搜索未返回结果")

    if result["status"] == ProcessingStatus.INVALID_INPUT:
        st.error("输入被判定为无效，请检查并重新输入。")
    elif result["status"] in [
        ProcessingStatus.UNIDENTIFIED,
        ProcessingStatus.UNVERIFIED,
    ]:
        st.warning("此结果可能需要进一步确认。")
    elif result["status"] == ProcessingStatus.VALID_INPUT:
        st.info(
            "输入被判定为有效，但未进行进一步处理。如需更高准确度，请考虑启用搜索和检索步骤。"
        )


def batch_processing(workflow: EntityVerificationWorkflow, entity_type: str):
    """
    处理批量实体名称标准化。

    Args:
        workflow (EntityVerificationWorkflow): 实体名称标准化工作流对象。
        entity_type (str): 实体类型。
    """
    uploaded_file = st.file_uploader(f"上传CSV文件（包含{entity_type}列）", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("预览上传的数据：")
        st.dataframe(df.head())

        # 添加并行处理任务数选择
        max_workers = st.slider(
            "选择并发任务数", min_value=1, max_value=10, value=5, step=1
        )

        if st.button("开始批量处理"):
            asyncio.run(process_batch(df, workflow, entity_type, max_workers))

    # 显示处理结果（如果有）
    if st.session_state.batch_results_df is not None:
        display_batch_results(st.session_state.batch_results_df, entity_type)


async def process_entity(
    entity_name: str, workflow: EntityVerificationWorkflow, session_id: str
) -> Dict[str, Any]:
    """
    处理单个实体的异步函数。

    Args:
        entity_name: 要处理的实体名称。
        workflow: 实体验证工作流对象。
        session_id: 会话ID。

    Returns:
        处理结果字典。
    """
    return await workflow.run(entity_name, session_id=session_id)


async def process_batch(
    df: pd.DataFrame,
    workflow: EntityVerificationWorkflow,
    entity_type: str,
    max_workers: int = 5,
):
    """
    异步处理批量实体数据，使用信号量控制并发。

    Args:
        df (pd.DataFrame): 包含实体名称的DataFrame。
        workflow (EntityVerificationWorkflow): 实体名称标准化工作流对象。
        entity_type (str): 实体类型。
        max_workers (int): 最大并发任务数，默认为5。
    """
    results = []
    progress_bar = st.progress(0)
    status_area = st.empty()
    total_entities = len(df)
    save_interval = 100

    semaphore = asyncio.Semaphore(max_workers)

    # 生成唯一的任务标识符
    task_id = time.strftime("%Y%m%d-%H%M%S")
    temp_dir = os.path.join("data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(
        temp_dir, f"batch_results_{entity_type}_{task_id}.csv"
    )

    async def process_with_semaphore(entity_name: str, index: int) -> Dict[str, Any]:
        async with semaphore:
            session_id = str(uuid4())
            result = await process_entity(entity_name, workflow, session_id)
            result["original_index"] = index
            return result

    async def save_results(current_results: List[Dict[str, Any]]):
        result_df = pd.DataFrame(current_results)
        result_df = result_df.sort_values("original_index").reset_index(drop=True)
        result_df = result_df.drop("original_index", axis=1)
        result_df.to_csv(temp_file_path, index=False, encoding="utf-8-sig")
        st.session_state.batch_results_df = result_df

    tasks = [
        process_with_semaphore(entity_name, i)
        for i, entity_name in enumerate(df.iloc[:, 0])
    ]

    async with aiohttp.ClientSession() as session:
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            progress = (i + 1) / total_entities
            progress_bar.progress(progress)

            status_message = (
                f"已处理: {i + 1}/{total_entities} - "
                f"最新: '{result['original_input']}' → '{result['final_entity_name']}' "
                f"(状态: {result['status'].value})"
            )
            status_area.info(status_message)

            # 每处理100个实体自动保存一次
            if (i + 1) % save_interval == 0:
                await save_results(results)
                st.info(f"已自动保存 {i + 1} 个处理结果")

    # 最后一次保存，确保所有结果都被保存
    await save_results(results)

    # 更新会话状态
    st.session_state.processing_complete = True
    st.success(f"批量处理完成！共处理 {len(results)} 条数据。")


def display_batch_results(result_df: pd.DataFrame, entity_type: str):
    st.success("批量处理完成！")

    # 计算统计数据
    status_counts = result_df["status"].value_counts()

    # 显示简化的结果统计
    st.subheader("处理结果统计")
    stats_df = pd.DataFrame(
        {
            "类别": [status.value for status in ProcessingStatus],
            "数量": [status_counts.get(status, 0) for status in ProcessingStatus],
        }
    )
    stats_df["占比"] = (stats_df["数量"] / len(result_df) * 100).round(2).astype(
        str
    ) + "%"
    st.table(stats_df.set_index("类别"))

    # 显示结果表格
    st.subheader("详细结果")
    display_columns = [
        "original_input",
        "final_entity_name",
        "status",
        "search_results",
    ]
    st.dataframe(result_df[display_columns])

    # 提供建议
    if (
        ProcessingStatus.UNVERIFIED in status_counts
        or ProcessingStatus.UNIDENTIFIED in status_counts
    ):
        st.warning(
            f"有 {status_counts.get(ProcessingStatus.UNVERIFIED, 0) + status_counts.get(ProcessingStatus.UNIDENTIFIED, 0)} 个实体未能完全验证。建议手动检查这些结果。"
        )
    if ProcessingStatus.VALID_INPUT in status_counts:
        st.info(
            f"有 {status_counts.get(ProcessingStatus.VALID_INPUT, 0)} 个实体仅进行了输入验证。如需更高准确度，请考虑启用完整识别流程。"
        )

    # 提供下载选项
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 下载处理结果",
        data=csv.encode("utf-8-sig"),
        file_name=f"processed_{entity_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


main()
