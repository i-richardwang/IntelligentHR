"""``SearchTools`` 的单元测试（隔离真实网络与 API）。

校验新搜索栈的结果清洗逻辑：
- Tavily（``langchain-tavily``）的 ``.ainvoke`` 返回 ``{"results": [...]}`` dict，
  需从 ``results`` 中提取 ``content``；
- DuckDuckGo（``ddgs``）返回 ``{title, href, body}`` 列表，需提取 ``body``。
"""

import asyncio

import pytest

import backend.data_processing.data_cleaning.search_tools as search_tools
from backend.data_processing.data_cleaning.search_tools import SearchTools


@pytest.fixture(autouse=True)
def _tavily_key(monkeypatch):
    # TavilySearch 构造需要 TAVILY_API_KEY（仅构造，不发起网络）
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")


def test_atavily_search_extracts_content_from_results():
    tools = SearchTools(max_results=3)

    class _FakeTavily:
        async def ainvoke(self, payload):
            assert payload == {"query": "q"}
            return {
                "results": [{"content": "a"}, {"title": "无 content"}, {"content": "b"}]
            }

    # TavilySearch 是 pydantic 模型不可 setattr 方法，直接替换整个工具对象
    tools.tavily_search_tool = _FakeTavily()
    assert asyncio.run(tools.atavily_search("q")) == str(["a", "b"])


def test_aduckduckgo_search_extracts_body(monkeypatch):
    class _FakeDDGS:
        def text(self, query, max_results=None):
            return [{"body": "x"}, {"title": "无 body"}, {"body": "y"}]

    monkeypatch.setattr(search_tools, "DDGS", lambda: _FakeDDGS())
    tools = SearchTools()
    assert asyncio.run(tools.aduckduckgo_search("q")) == str(["x", "y"])
