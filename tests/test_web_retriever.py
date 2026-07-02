"""``backend.ai_research.web_retriever`` 的单元测试。

聚焦回填的抓取健壮性逻辑：跳过 PDF、HTTP 错误返回空、按最小内容
长度过滤，以及 Tavily 密钥缺失时的显式报错。网络与解析均以桩替身
隔离，不发起真实请求。
"""

import pytest
import requests

from backend.ai_research.web_retriever import (
    BeautifulSoupScraper,
    Scraper,
    SearchAPIRetriever,
    TavilySearch,
    TavilyAPIError,
    get_retriever,
    get_default_retriever,
)


class FakeResponse:
    def __init__(self, content=b"", headers=None, encoding="utf-8", status_ok=True):
        self.content = content
        self.headers = headers or {}
        self.encoding = encoding
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise requests.exceptions.HTTPError("500 Server Error")


class FakeSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, timeout=None):
        return self._response


def test_scrape_skips_pdf():
    resp = FakeResponse(
        content=b"%PDF-1.4 binary", headers={"Content-Type": "application/pdf"}
    )
    scraper = BeautifulSoupScraper("http://x/doc.pdf", session=FakeSession(resp))
    assert scraper.scrape() == ""


def test_scrape_http_error_returns_empty():
    resp = FakeResponse(status_ok=False, headers={"Content-Type": "text/html"})
    scraper = BeautifulSoupScraper("http://x", session=FakeSession(resp))
    assert scraper.scrape() == ""


class _ShortScraper:
    def __init__(self, link, session=None, config=None):
        self.link = link

    def scrape(self):
        return "tiny"


class _LongScraper:
    def __init__(self, link, session=None, config=None):
        self.link = link

    def scrape(self):
        return "x" * 500


def test_scraper_filters_short_content():
    # 默认 min_content_length=100，"tiny" 会被过滤，run() 丢弃 raw_content 为 None 的项
    scraper = Scraper(["http://a"], "ua", scraper_class=_ShortScraper)
    assert scraper.run() == []


def test_scraper_keeps_long_content():
    scraper = Scraper(["http://a"], "ua", scraper_class=_LongScraper)
    out = scraper.run()
    assert len(out) == 1
    assert out[0]["url"] == "http://a"
    assert out[0]["raw_content"] == "x" * 500


def test_tavily_missing_key_raises(monkeypatch):
    # 密钥缺失属配置错误，应在构造时直接抛出，而非静默回退
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyAPIError):
        TavilySearch("some query")


def test_get_retriever_lookup():
    assert get_retriever("tavily") is TavilySearch
    assert get_retriever("unknown") is None
    assert get_default_retriever() is TavilySearch


def test_search_api_retriever_invoke():
    # 回归：langchain 1.x 的 BaseRetriever 要求实现 _get_relevant_documents，
    # 否则连实例化都会失败、.invoke 也无法工作
    retriever = SearchAPIRetriever()
    retriever.pages = [
        {"raw_content": "hello world", "title": "T", "url": "http://u"}
    ]
    docs = retriever.invoke("q")
    assert len(docs) == 1
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == "http://u"
