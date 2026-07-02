"""
SQL助手的异步执行模块。
将同步的图执行逻辑放到线程池中运行，避免阻塞 FastAPI 的事件循环。
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, Optional

from backend.sql_assistant.graph.assistant_graph import run_sql_assistant

logger = logging.getLogger(__name__)

# 线程池执行器。
# run_sql_assistant 是完全同步的实现，内部包含多次阻塞式的 LLM 与数据库调用。
# 若直接在异步接口中调用，会占用事件循环导致并发请求相互排队；
# 放入线程池执行可让多个请求真正并发处理。
thread_pool = ThreadPoolExecutor(max_workers=3)


async def run_sql_assistant_async(
    query: str,
    thread_id: Optional[str] = None,
    checkpoint_saver: Optional[Any] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """在线程池中异步执行 SQL 助手，避免阻塞事件循环。

    Args:
        query: 用户的查询文本
        thread_id: 会话ID
        checkpoint_saver: 状态保存器实例
        user_id: 用户ID,用于权限控制

    Returns:
        Dict[str, Any]: 处理结果字典
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_pool,
        partial(
            run_sql_assistant,
            query=query,
            thread_id=thread_id,
            checkpoint_saver=checkpoint_saver,
            user_id=user_id,
        ),
    )
