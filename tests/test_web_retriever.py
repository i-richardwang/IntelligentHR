"""``backend.ai_research.web_retriever`` 的单元测试。

该模块此前在异步研究管线里用同步 ``requests`` + ``ThreadPoolExecutor``，会阻塞事件循环。
现改为 httpx 异步客户端 + ``asyncio.gather`` 并发抓取、``AsyncTavilyClient`` 异步搜索、
``ddgs`` 经 ``asyncio.to_thread`` 回退。此处验证抓取健壮性（跳过 PDF、HTTP 错误返回空、
按最小内容长度过滤）、搜索与回退，以及 Tavily 密钥缺失时的显式报错。

沿用本仓库约定：无 pytest-asyncio，异步用例经 ``asyncio.run`` 驱动，网络与解析均以
``unittest.mock`` 桩替身隔离，不发起真实请求。
"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import httpx
import pytest

from backend.ai_research.web_retriever import (
    BeautifulSoupScraper,
    Scraper,
    SearchAPIRetriever,
    TavilySearch,
    TavilyAPIError,
    scrape_urls,
    get_retriever,
    get_default_retriever,
)


class _Cfg:
    """抓取配置桩。"""

    user_agent = "test-agent"
    scrape_timeout = 8
    scrape_max_workers = 5
    min_content_length = 5


def _html_response(body_html: str, content_type: str = "text/html"):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.headers = {"Content-Type": content_type}
    resp.content = f"<html><body>{body_html}</body></html>".encode("utf-8")
    resp.encoding = "utf-8"
    return resp


# ---------------- TavilySearch（异步搜索 + DuckDuckGo 回退）----------------


def test_tavily_search_returns_results():
    ts = TavilySearch("查询", headers={"tavily_api_key": "k"})
    ts.client = MagicMock(
        search=AsyncMock(return_value={"results": [{"url": "u1", "content": "c1"}]})
    )
    out = asyncio.run(ts.search(max_results=3))
    assert out == [{"href": "u1", "body": "c1"}]
    ts.client.search.assert_awaited_once()


def test_tavily_search_falls_back_to_ddg_when_empty():
    ts = TavilySearch("查询", headers={"tavily_api_key": "k"})
    ts.client = MagicMock(search=AsyncMock(return_value={"results": []}))
    ddg_hit = [{"title": "t", "href": "h", "body": "b"}]
    fake_ddgs = MagicMock()
    fake_ddgs.return_value.text.return_value = ddg_hit
    with patch("backend.ai_research.web_retriever.DDGS", fake_ddgs):
        out = asyncio.run(ts.search(max_results=3))
    assert out == ddg_hit


def test_tavily_search_falls_back_on_exception_then_empty():
    ts = TavilySearch("查询", headers={"tavily_api_key": "k"})
    ts.client = MagicMock(search=AsyncMock(side_effect=RuntimeError("boom")))
    fake_ddgs = MagicMock()
    fake_ddgs.return_value.text.return_value = []
    with patch("backend.ai_research.web_retriever.DDGS", fake_ddgs):
        out = asyncio.run(ts.search())
    # Tavily 异常 + DDG 无结果 -> 空列表
    assert out == []


def test_tavily_missing_key_raises(monkeypatch):
    # 密钥缺失属配置错误，应在构造时直接抛出，而非静默回退
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyAPIError):
        TavilySearch("some query")


# ---------------- BeautifulSoupScraper（异步抓取）----------------


def test_scrape_parses_html():
    scraper = BeautifulSoupScraper("http://x", _Cfg())
    client = MagicMock()
    client.get = AsyncMock(return_value=_html_response("<p>hello world content</p>"))
    text = asyncio.run(scraper.scrape(client))
    assert "hello world content" in text
    client.get.assert_awaited_once_with("http://x")


def test_scrape_skips_pdf():
    scraper = BeautifulSoupScraper("http://x/doc.pdf", _Cfg())
    client = MagicMock()
    client.get = AsyncMock(
        return_value=_html_response("<p>x</p>", content_type="application/pdf")
    )
    assert asyncio.run(scraper.scrape(client)) == ""


def test_scrape_swallows_http_error():
    scraper = BeautifulSoupScraper("http://x", _Cfg())
    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.HTTPError("网络错误"))
    # httpx.HTTPError 被捕获，返回空串（不冒泡）
    assert asyncio.run(scraper.scrape(client)) == ""


# ---------------- Scraper / scrape_urls（并发抓取 + 过滤）----------------


def _patched_async_client(get_side_effect):
    """构造一个可作为 async 上下文管理器的 httpx.AsyncClient 桩。"""
    client = MagicMock()
    client.get = AsyncMock(side_effect=get_side_effect)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return client, ctx


def test_scrape_urls_concurrent_and_filters_short():
    long_resp = _html_response("<p>this is long enough content</p>")
    short_resp = _html_response("<p>x</p>")  # 短于 min_content_length -> 过滤
    client, ctx = _patched_async_client([long_resp, short_resp])
    with patch(
        "backend.ai_research.web_retriever.httpx.AsyncClient", return_value=ctx
    ):
        results = asyncio.run(scrape_urls(["http://a", "http://b"], _Cfg()))
    # 仅保留内容达标的那条
    assert len(results) == 1
    assert results[0]["url"] == "http://a"
    assert client.get.await_count == 2


def test_scraper_run_all_short_returns_empty():
    client, ctx = _patched_async_client([_html_response("<p>x</p>")])
    with patch(
        "backend.ai_research.web_retriever.httpx.AsyncClient", return_value=ctx
    ):
        out = asyncio.run(Scraper(["http://a"], "ua", config=_Cfg()).run())
    assert out == []


# ---------------- 检索器查找 / SearchAPIRetriever 契约 ----------------


def test_get_retriever_lookup():
    assert get_retriever("tavily") is TavilySearch
    assert get_retriever("unknown") is None
    assert get_default_retriever() is TavilySearch


def test_search_api_retriever_invoke():
    # 回归：langchain 1.x 的 BaseRetriever 要求实现 _get_relevant_documents，
    # 否则连实例化都会失败、.invoke 也无法工作
    retriever = SearchAPIRetriever()
    retriever.pages = [{"raw_content": "hello world", "title": "T", "url": "http://u"}]
    docs = retriever.invoke("q")
    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == "http://u"
