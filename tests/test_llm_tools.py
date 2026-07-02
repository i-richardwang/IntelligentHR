"""``utils.llm_tools.LanguageModelChain`` 的单元测试。

回归保护：langchain 1.x 的聊天模型是 ``Runnable`` 而非 ``callable``
（``callable(ChatOpenAI())`` 返回 False），LanguageModelChain 的参数
校验必须接受它，否则全站基于 LanguageModelChain 的 LLM 功能都会崩。
"""

import pytest
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

from utils.llm_tools import LanguageModelChain


class _Output(BaseModel):
    answer: str


def test_accepts_chat_model():
    # langchain 1.x 下 callable(ChatOpenAI 实例) == False，但它是合法 Runnable
    model = ChatOpenAI(api_key="x", base_url="http://localhost:1", model="gpt-4o")
    chain = LanguageModelChain(_Output, "系统消息", "用户消息", model)
    assert chain() is not None


def test_accepts_plain_runnable():
    chain = LanguageModelChain(_Output, "sys", "user", RunnableLambda(lambda x: x))
    assert chain() is not None


def test_rejects_non_runnable():
    with pytest.raises(ValueError):
        LanguageModelChain(_Output, "sys", "user", None)
