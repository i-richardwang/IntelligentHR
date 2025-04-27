"""
简历信息提取核心逻辑模块。

本模块包含了简历信息提取的主要功能函数，包括个人信息和教育背景提取、
工作和项目经历提取，以及简历概述生成。
"""

import os
import asyncio
import hashlib
import logging
import uuid
from typing import Dict, Any

from langfuse.callback import CallbackHandler

from utils.llm_tools import LanguageModelChain, init_language_model
from backend.resume_management.extractor.resume_data_models import (
    ResumePersonalEducation,
    ResumeWorkProject,
    Summary,
)
from backend.resume_management.extractor.resume_extraction_prompts import (
    PERSONAL_EDUCATION_SYSTEM_MESSAGE,
    PERSONAL_EDUCATION_HUMAN_MESSAGE,
    WORK_PROJECT_SYSTEM_MESSAGE,
    WORK_PROJECT_HUMAN_MESSAGE,
    RESUME_SUMMARY_SYSTEM_MESSAGE,
    RESUME_SUMMARY_HUMAN_MESSAGE,
)
from backend.resume_management.storage.resume_vector_storage import (
    store_resume_in_milvus,
)
from backend.resume_management.storage.resume_sql_storage import (
    init_all_tables,
    store_full_resume,
)

# 配置日志
logging.basicConfig(level=logging.INFO)

# 初始化语言模型
language_model = init_language_model(
    provider=os.getenv("FAST_LLM_PROVIDER"), model_name=os.getenv("FAST_LLM_MODEL")
)


async def extract_personal_education(
    resume_content: str, session_id: str
) -> Dict[str, Any]:
    """
    提取简历中的个人信息和教育背景。

    Args:
        resume_content (str): 简历内容。
        session_id (str): 会话ID。

    Returns:
        Dict[str, Any]: 包含个人信息和教育背景的字典。
    """
    langfuse_handler = create_langfuse_handler(session_id, "extract_personal_education")
    extractor = LanguageModelChain(
        ResumePersonalEducation,
        PERSONAL_EDUCATION_SYSTEM_MESSAGE,
        PERSONAL_EDUCATION_HUMAN_MESSAGE,
        language_model,
    )()
    return await extractor.ainvoke(
        {"raw_resume_content": resume_content}, config={"callbacks": [langfuse_handler]}
    )


async def extract_work_project(resume_content: str, session_id: str) -> Dict[str, Any]:
    """
    提取简历中的工作和项目经历。

    Args:
        resume_content (str): 简历内容。
        session_id (str): 会话ID。

    Returns:
        Dict[str, Any]: 包含工作和项目经历的字典。
    """
    langfuse_handler = create_langfuse_handler(session_id, "extract_work_project")
    extractor = LanguageModelChain(
        ResumeWorkProject,
        WORK_PROJECT_SYSTEM_MESSAGE,
        WORK_PROJECT_HUMAN_MESSAGE,
        language_model,
    )()
    return await extractor.ainvoke(
        {"raw_resume_content": resume_content}, config={"callbacks": [langfuse_handler]}
    )


async def generate_resume_summary(
    resume_content: str, session_id: str
) -> Dict[str, Any]:
    """
    生成简历概述。

    Args:
        resume_content (str): 简历内容。
        session_id (str): 会话ID。

    Returns:
        Dict[str, Any]: 包含简历概述的字典。
    """
    langfuse_handler = create_langfuse_handler(session_id, "generate_resume_summary")
    summarizer = LanguageModelChain(
        Summary,
        RESUME_SUMMARY_SYSTEM_MESSAGE,
        RESUME_SUMMARY_HUMAN_MESSAGE,
        language_model,
    )()
    return await summarizer.ainvoke(
        {"raw_resume_content": resume_content}, config={"callbacks": [langfuse_handler]}
    )


async def process_resume(
    resume_content: str,
    resume_id: str,
    session_id: str = None,
    file_type: str = "url",
    file_or_url: str = "",
) -> Dict[str, Any]:
    """
    处理简历，提取所有相关信息。

    Args:
        resume_content (str): 简历内容。
        resume_id (str): 简历ID。
        session_id (str, optional): 会话ID。如果未提供，将生成新的UUID。
        file_type (str, optional): 文件类型。默认为"url"。
        file_or_url (str, optional): 文件路径或URL。默认为空字符串。

    Returns:
        Dict[str, Any]: 包含完整简历信息的字典。
    """
    session_id = session_id or str(uuid.uuid4())

    try:
        personal_education_task = extract_personal_education(resume_content, session_id)
        work_project_task = extract_work_project(resume_content, session_id)
        summary_task = generate_resume_summary(resume_content, session_id)

        personal_education, work_project, summary = await asyncio.gather(
            personal_education_task, work_project_task, summary_task
        )

        complete_resume = {
            "id": resume_id,
            "personal_info": personal_education.get("personal_info", {}),
            "education": personal_education.get("education", []),
            "work_experiences": work_project.get("work_experiences", []),
            "project_experiences": work_project.get("project_experiences", []),
            "summary": summary.get("summary", {}),
            "characteristics": summary.get("summary", {}).get("characteristics", ""),
            "experience_summary": summary.get("summary", {}).get("experience", ""),
            "skills_overview": summary.get("summary", {}).get("skills_overview", ""),
            "resume_format": file_type,
            "file_or_url": file_or_url,
        }

        return complete_resume
    except Exception as e:
        logging.error(f"处理简历时出错: {str(e)}")
        return {"error": f"处理简历时出错: {str(e)}"}


def calculate_resume_hash(resume_content: str) -> str:
    """
    计算简历内容的哈希值。

    Args:
        resume_content (str): 简历内容文本。

    Returns:
        str: 简历内容的MD5哈希值。
    """
    return hashlib.md5(resume_content.encode()).hexdigest()


def store_resume(resume_data: Dict[str, Any]) -> bool:
    """
    将处理后的简历数据存储到Milvus数据库和SQLite数据库中。

    Args:
        resume_data (Dict[str, Any]): 处理后的简历数据。

    Returns:
        bool: 存储操作是否成功。
    """
    try:
        # 存储到Milvus
        store_resume_in_milvus(resume_data)

        # 存储到SQLite
        init_all_tables()
        store_full_resume(resume_data)
        return True
    except Exception as e:
        logging.error(f"存储简历数据时出错: {str(e)}")
        return False


def create_langfuse_handler(session_id: str, step: str) -> CallbackHandler:
    """
    创建Langfuse回调处理器。

    Args:
        session_id (str): 会话ID。
        step (str): 当前步骤。

    Returns:
        CallbackHandler: Langfuse回调处理器实例。
    """
    return CallbackHandler(
        tags=["resume_extraction"], session_id=session_id, metadata={"step": step}
    )
