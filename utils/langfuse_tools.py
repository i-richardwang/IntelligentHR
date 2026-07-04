"""
Langfuse 监控工具。

统一封装 Langfuse 与 langchain / langgraph 的接入：通过 ``get_langfuse_config``
构造带回调的 invoke config，并把 ``session_id`` / ``tags`` / ``user_id`` 等 trace
属性写入 config 的 ``metadata``——以 ``langfuse_`` 前缀的键传递（Langfuse 约定）。

用法：

    from utils.langfuse_tools import get_langfuse_config

    config = get_langfuse_config(
        session_id=session_id,
        tags=["content_analysis"],
        metadata={"step": step},
    )
    result = chain.invoke({...}, config=config)

若未配置 Langfuse 环境变量（``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``），
则优雅降级：返回不含回调的 config，业务调用照常执行，只是不上报监控。
"""

import os
import uuid
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LangfuseManager:
    """Langfuse 监控管理器（单例），负责初始化全局客户端。"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._enabled = self._init_client()
        self._initialized = True

    def _init_client(self) -> bool:
        """
        按 IHR 的环境变量约定初始化 Langfuse 全局单例客户端。

        Returns:
            bool: 配置完整且初始化成功返回 True，否则 False（监控禁用）。
        """
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

        if not (public_key and secret_key):
            logger.warning("⚠️ 未检测到完整的 Langfuse 配置，监控已禁用。")
            return False

        try:
            from langfuse import Langfuse, get_client


            Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            _ = get_client()  # 触发全局客户端创建
            logger.info("✅ Langfuse 初始化成功（host=%s）。", host)
            return True
        except Exception as e:  # noqa: BLE001 - 监控失败不应阻断业务
            logger.error("❌ Langfuse 初始化失败：%s", e)
            return False

    @property
    def is_enabled(self) -> bool:
        """监控是否已启用。"""
        return self._enabled

    def get_callback_handler(self):
        """
        返回一个 Langfuse ``CallbackHandler``；未启用或创建失败时返回 None。

        CallbackHandler 不在构造函数接收 session_id/tags/metadata，这些属性通过
        invoke config 的 metadata 传递，见 ``get_langfuse_config``。
        """
        if not self._enabled:
            return None
        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler()
        except Exception as e:  # noqa: BLE001
            logger.error("❌ 创建 Langfuse CallbackHandler 失败：%s", e)
            return None


# 全局单例
langfuse_manager = LangfuseManager()


def get_langfuse_config(
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    existing_config: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    构造带 Langfuse 回调的 langchain / langgraph invoke config。

    Langfuse 约定：``session_id`` / ``tags`` / ``user_id`` 通过 config 的 ``metadata``
    里带 ``langfuse_`` 前缀的键传递；其余自定义 metadata（如 ``{"step": ...}``）
    原样并入 ``metadata``。

    Args:
        session_id: 会话 ID。
        tags: trace 标签列表。
        metadata: 额外自定义元数据（如 ``{"step": "..."}``）。
        existing_config: 已有的 config（会被浅拷贝后合并，用于保留如
            langgraph 的 ``configurable`` 等键）。
        user_id: 可选用户 ID。

    Returns:
        Dict[str, Any]: 可直接传给 ``chain.invoke(..., config=...)`` 的字典。
        未启用监控时，返回不含回调的原 config（业务照常执行）。
    """
    config: Dict[str, Any] = dict(existing_config) if existing_config else {}

    handler = langfuse_manager.get_callback_handler()
    if handler is None:
        # 监控未启用：保持 config 原样，不注入 langfuse_* 键。
        return config

    callbacks = list(config.get("callbacks", []))
    callbacks.append(handler)
    config["callbacks"] = callbacks

    # Langfuse v4 约定：propagated metadata 为 dict[str, str]，非字符串值会被强制
    # 转换、超过 200 字符的值会被丢弃并告警。这里主动把 session_id/user_id 及自定义
    # metadata 的值统一转成字符串，规避隐式转换告警；langfuse_tags 是特殊键，保持
    # list 形态由集成层消费。
    md: Dict[str, Any] = dict(config.get("metadata", {}))
    if session_id is not None:
        md["langfuse_session_id"] = str(session_id)
    if tags:
        md["langfuse_tags"] = list(tags)
    if user_id is not None:
        md["langfuse_user_id"] = str(user_id)
    if metadata:
        md.update(
            {k: v if isinstance(v, str) else str(v) for k, v in metadata.items()}
        )
    config["metadata"] = md

    return config


def generate_session_id() -> str:
    """生成新的会话 ID（UUID 字符串）。"""
    return str(uuid.uuid4())
