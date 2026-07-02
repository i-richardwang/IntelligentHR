"""
SQL助手的API入口模块。
提供HTTP异步接口供外部调用SQL助手服务。
"""

import os
import asyncio
from typing import Optional, Dict, Any, NamedTuple
import httpx
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel
from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from backend.sql_assistant.async_executor import run_sql_assistant_async
from backend.sql_assistant.utils.user_mapper import UserMapper
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

import logging

# 设置日志记录器
logger = logging.getLogger(__name__)

# 设置LLM缓存
set_llm_cache(SQLiteCache(database_path="./data/llm_cache/langchain.db"))

# FastAPI应用
app = FastAPI(title="SQL助手API")

# 配置


class Config:
    USER_AUTH_ENABLED = os.getenv("USER_AUTH_ENABLED", "").lower() == "true"


class ChatResponse(BaseModel):
    text: str
    session_id: Optional[str] = None


# 全局变量
checkpoint_saver = MemorySaver()
user_mapper = UserMapper()


@app.post("/api/sql-assistant", response_model=ChatResponse)
async def process_query(request: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    """处理SQL查询请求"""
    try:
        text = request.get("text", "")
        username = request.get("username", "anonymous")
        session_id = request.get("session_id", None)

        # 根据环境变量决定是否进行用户权限控制
        user_id = None
        if Config.USER_AUTH_ENABLED:
            user_id = user_mapper.get_user_id(username)
            if user_id is None and username != "anonymous":
                raise HTTPException(
                    status_code=404,
                    detail=f"抱歉，您暂时没有访问权限。请联系数据团队为您开通相关权限。",
                )

        # 运行SQL助手（在线程池中执行，避免阻塞事件循环）
        result = await run_sql_assistant_async(
            query=text,
            thread_id=session_id,
            checkpoint_saver=checkpoint_saver,
            user_id=user_id,
        )

        # 获取最后一条助手消息
        messages = result.get("messages", [])
        if not messages:
            raise ValueError("未获取到助手回复")

        last_message = messages[-1].content

        return ChatResponse(
            text=last_message,
            session_id=result.get("thread_id", ""),
        )

    except HTTPException as he:
        # 直接重新抛出 HTTPException，保留原始的状态码和错误信息
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="抱歉，系统处理您的请求时遇到了问题。请稍后重试，如果问题持续存在，请联系数据团队寻求帮助。",
        )


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}
