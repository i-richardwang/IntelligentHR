import logging
import os
import uuid
from typing import Dict, Any

from langfuse.callback import CallbackHandler
from pydantic import BaseModel, Field

from utils.llm_tools import init_language_model, LanguageModelChain

logger = logging.getLogger(__name__)


class TranslatedText(BaseModel):
    """表示翻译后的文本的数据模型。"""

    translated_text: str = Field(..., description="翻译成中文的文本内容")


SYSTEM_MESSAGE = """
你是一位精通多语言的翻译专家。你的任务是将给定的{text_topic}文本准确翻译成中文。请遵循以下指南：

1. 翻译要求：
   - 仔细阅读每条文本，理解其核心内容和语境。
   - 将文本准确翻译成中文，保持原意不变。
   - 确保翻译后的文本通顺、自然，符合中文表达习惯。
   - 如遇专业术语或特定概念，请尽可能找到恰当的中文对应表述。

2. 输出格式：
   - 对每条文本，输出对应的中文翻译。
   - 忽略原始文本中的特殊格式，按照一段话的形式输出翻译结果，不要包含特殊字符。

请确保翻译的准确性和一致性，不要遗漏任何内容。
"""

HUMAN_MESSAGE_TEMPLATE = """
请将以下{text_topic}文本翻译成中文。

```
{text_to_translate}
```

请按照系统消息中的指南进行翻译，并以指定的JSON格式输出结果，但不要在输出中重复json schema。
"""


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
        tags=["translation"], session_id=session_id, metadata={"step": step}
    )


class Translator:
    """翻译器类，用于处理文本翻译任务。"""

    def __init__(self, temperature: float = 0.0):
        """
        初始化翻译器。

        Args:
            temperature (float): 语言模型的温度参数，控制输出的随机性。
        """
        self.language_model = init_language_model(
            temperature=temperature,
            provider=os.getenv("FAST_LLM_PROVIDER"),
            model_name=os.getenv("FAST_LLM_MODEL"),
        )
        self.translation_chain = LanguageModelChain(
            TranslatedText, SYSTEM_MESSAGE, HUMAN_MESSAGE_TEMPLATE, self.language_model
        )()

    async def translate(
        self, text: str, text_topic: str, session_id: str = None
    ) -> str:
        """
        异步翻译单个文本。

        Args:
            text (str): 要翻译的文本。
            text_topic (str): 文本主题，用于上下文理解。
            session_id (str, optional): 会话ID，用于Langfuse监控。

        Returns:
            str: 翻译后的文本。

        Raises:
            ValueError: 当翻译结果格式不正确时抛出。
            Exception: 当翻译过程中发生其他错误时抛出。
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        try:
            langfuse_handler = create_langfuse_handler(session_id, "translate")
            result = await self.translation_chain.ainvoke(
                {"text_to_translate": text, "text_topic": text_topic},
                config={"callbacks": [langfuse_handler]},
            )
            self._validate_translation_result(result)
            return result["translated_text"]
        except ValueError as ve:
            logger.error(f"翻译结果格式不正确: {ve}")
            raise
        except Exception as e:
            logger.error(f"翻译过程中发生错误: {e}", exc_info=True)
            raise

    @staticmethod
    def _validate_translation_result(result: Dict[str, Any]) -> None:
        """
        验证翻译结果的格式。

        Args:
            result (Dict[str, Any]): 翻译结果字典。

        Raises:
            ValueError: 当结果格式不正确时抛出。
        """
        if not isinstance(result, dict) or "translated_text" not in result:
            raise ValueError("翻译结果格式不正确")
