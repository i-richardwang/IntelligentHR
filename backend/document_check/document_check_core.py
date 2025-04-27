import os
import asyncio
import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langfuse.callback import CallbackHandler

from utils.llm_tools import LanguageModelChain, init_language_model
from backend.document_check.document_check_prompts import (
    DOCUMENT_CHECK_SYSTEM_MESSAGE,
    DOCUMENT_CHECK_HUMAN_MESSAGE,
)


# 数据模型
class CorrectionItem(BaseModel):
    element_id: str = Field(..., description="元素ID")
    original_text: str = Field(..., description="原始文本")
    suggestion: str = Field(..., description="修改建议")
    correction_reason: str = Field(..., description="修改理由，解释错误的原因")


class DocumentCheck(BaseModel):
    corrections: List[CorrectionItem] = Field(
        default=[], description="检测到的需要修改的内容列表(不需要返回无需修改的内容)"
    )


# 初始化语言模型
language_model = init_language_model(
    provider=os.getenv("SMART_LLM_PROVIDER"), model_name=os.getenv("SMART_LLM_MODEL")
)

# 创建大模型调用链
document_checker = LanguageModelChain(
    DocumentCheck,
    DOCUMENT_CHECK_SYSTEM_MESSAGE,
    DOCUMENT_CHECK_HUMAN_MESSAGE,
    language_model,
)()


def create_langfuse_handler(session_id: str, step: str) -> CallbackHandler:
    return CallbackHandler(
        tags=["document_check"], session_id=session_id, metadata={"step": step}
    )


async def check_page(
    page_elements: List[dict], session_id: str, page_number: int
) -> Dict[str, Any]:
    langfuse_handler = create_langfuse_handler(session_id, f"check_page_{page_number}")

    formatted_content = "\n".join(
        [
            f"(ID: {element['element_id']}): {element['text']}"
            for element in page_elements
        ]
    )

    result = await document_checker.ainvoke(
        {"document_content": formatted_content},
        config={"callbacks": [langfuse_handler]},
    )

    return {"page_number": page_number, **result}


async def check_document(
    document_content: List[dict], session_id: str
) -> List[Dict[str, Any]]:
    pages = {}
    for element in document_content:
        page_number = element["metadata"]["page_number"]
        if page_number not in pages:
            pages[page_number] = []
        pages[page_number].append(element)

    async def process_pages(page_items):
        tasks = []
        semaphore = asyncio.Semaphore(3)  # 限制同时处理的页面数为3

        async def check_with_semaphore(page_number, page_elements):
            async with semaphore:
                return await check_page(page_elements, session_id, page_number)

        for page_number, page_elements in page_items:
            task = asyncio.create_task(check_with_semaphore(page_number, page_elements))
            tasks.append(task)

        return await asyncio.gather(*tasks)

    results = await process_pages(pages.items())
    return results


async def process_document(document_content: List[dict]) -> List[Dict[str, Any]]:
    session_id = str(uuid.uuid4())
    check_results = await check_document(document_content, session_id)
    return check_results
