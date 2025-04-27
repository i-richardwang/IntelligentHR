# backend/ai_research/embedding_service.py
import os
from typing import List, Optional
from langchain.embeddings.base import Embeddings
from utils.llm_tools import CustomEmbeddings


class Memory:
    """内存类，用于管理和获取嵌入模型"""

    def __init__(
        self, embedding_provider: str, headers: Optional[dict] = None, **kwargs
    ):
        """
        初始化内存类

        :param embedding_provider: 嵌入提供者的名称
        :param headers: 可选的HTTP头部信息
        :param kwargs: 其他可选参数
        :raises ValueError: 当嵌入提供者不支持时抛出
        """
        self._embeddings = None
        headers = headers or {}

        if embedding_provider == "openai":
            self._embeddings = CustomEmbeddings(
                api_key=os.getenv("EMBEDDING_API_KEY", ""),
                api_url=os.getenv("EMBEDDING_API_BASE", ""),
                model=os.getenv("EMBEDDING_MODEL", ""),
            )
        else:
            raise ValueError("不支持的嵌入提供者。")

    def get_embeddings(self) -> Embeddings:
        """
        获取嵌入模型

        :return: 嵌入模型实例
        """
        return self._embeddings
