import os
import uuid
import asyncio
from typing import List
from utils.llm_tools import init_language_model, LanguageModelChain
from backend_demo.text_processing.classification.content_analysis_core import (
    ContentAnalysisResult,
    ContentAnalysisInput,
    CONTENT_ANALYSIS_SYSTEM_PROMPT,
    CONTENT_ANALYSIS_HUMAN_PROMPT,
)
from langfuse.callback import CallbackHandler


class TextContentAnalysisWorkflow:
    """文本内容分析工作流程类"""

    def __init__(self):
        """初始化文本内容分析工作流程"""
        self.language_model = init_language_model(
            provider=os.getenv("SMART_LLM_PROVIDER"),
            model_name=os.getenv("SMART_LLM_MODEL"),
        )
        self.analysis_chain = LanguageModelChain(
            ContentAnalysisResult,
            CONTENT_ANALYSIS_SYSTEM_PROMPT,
            CONTENT_ANALYSIS_HUMAN_PROMPT,
            self.language_model,
        )()

    def analyze_text(
        self, input_data: ContentAnalysisInput, session_id: str = None
    ) -> ContentAnalysisResult:
        """
        执行单个文本内容分析任务

        Args:
            input_data: 包含待分析文本和上下文的输入数据
            session_id: 可选的会话ID

        Returns:
            分析结果，包含有效性、情感倾向和敏感信息标识
        """
        session_id = session_id or str(uuid.uuid4())
        langfuse_handler = create_langfuse_handler(
            session_id, "content_analysis"
        )

        result = self.analysis_chain.invoke(
            {
                "text": input_data.text,
                "context": input_data.context,
            },
            config={"callbacks": [langfuse_handler]},
        )
        return ContentAnalysisResult(**result)

    def batch_analyze(
        self, texts: List[str], context: str, session_id: str
    ) -> List[ContentAnalysisResult]:
        """
        批量执行文本内容分析任务

        Args:
            texts: 待分析的文本列表
            context: 文本的上下文或主题
            session_id: 批量任务的会话ID

        Returns:
            分析结果列表
        """
        return [
            self.analyze_text(
                ContentAnalysisInput(text=text, context=context), session_id
            )
            for text in texts
        ]

    async def async_analyze_text(
        self, input_data: ContentAnalysisInput, session_id: str = None
    ) -> ContentAnalysisResult:
        """
        异步执行单个文本内容分析任务

        Args:
            input_data: 包含待分析文本和上下文的输入数据
            session_id: 可选的会话ID

        Returns:
            分析结果，包含有效性、情感倾向和敏感信息标识
        """
        session_id = session_id or str(uuid.uuid4())
        langfuse_handler = create_langfuse_handler(
            session_id, "content_analysis"
        )

        result = await self.analysis_chain.ainvoke(
            {
                "text": input_data.text,
                "context": input_data.context,
            },
            config={"callbacks": [langfuse_handler]},
        )
        return ContentAnalysisResult(**result)

    async def async_batch_analyze(
        self, texts: List[str], context: str, session_id: str, max_concurrency: int = 3
    ) -> List[ContentAnalysisResult]:
        """
        异步批量执行文本内容分析任务

        Args:
            texts: 待分析的文本列表
            context: 文本的上下文或主题
            session_id: 批量任务的会话ID
            max_concurrency: 最大并发数

        Returns:
            分析结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def analyze_with_semaphore(text):
            async with semaphore:
                input_data = ContentAnalysisInput(text=text, context=context)
                return await self.async_analyze_text(input_data, session_id)

        tasks = [analyze_with_semaphore(text) for text in texts]
        return await asyncio.gather(*tasks)


def create_langfuse_handler(session_id: str, step: str) -> CallbackHandler:
    """
    创建Langfuse回调处理器

    Args:
        session_id: 会话ID
        step: 处理步骤

    Returns:
        Langfuse回调处理器
    """
    return CallbackHandler(
        tags=["content_analysis"],
        session_id=session_id,
        metadata={"step": step},
    )
