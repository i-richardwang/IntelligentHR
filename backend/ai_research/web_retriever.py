# backend/ai_research/web_retriever.py

import os
import asyncio
import logging
from typing import List, Dict, Any
import httpx
from bs4 import BeautifulSoup

from tavily import AsyncTavilyClient
from ddgs import DDGS

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class TavilyAPIError(Exception):
    """Tavily API 相关错误（如缺少密钥、配置问题）。"""

    pass


class ScrapingError(Exception):
    """网页抓取相关错误。"""

    pass


class TavilySearch:
    """Tavily搜索客户端"""

    def __init__(
        self, query: str, headers: Dict[str, str] = None, topic: str = "general"
    ):
        """
        初始化Tavily搜索客户端

        :param query: 搜索查询
        :param headers: 请求头
        :param topic: 搜索主题
        """
        self.query = query
        self.headers = headers or {}
        self.api_key = self._get_api_key()
        self.client = AsyncTavilyClient(self.api_key)
        self.topic = topic

    def _get_api_key(self) -> str:
        """
        获取Tavily API密钥

        :return: API密钥
        :raises TavilyAPIError: 如果未找到API密钥
        """
        api_key = self.headers.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise TavilyAPIError(
                "未找到Tavily API密钥。请设置TAVILY_API_KEY环境变量。"
            )
        return api_key

    async def search(self, max_results: int = 7) -> List[Dict[str, str]]:
        """
        执行搜索（异步）

        Tavily 无结果或调用出错时，回退到 DuckDuckGo；两者都失败时返回空列表。
        （API 密钥缺失属配置错误，会在客户端构造时以 TavilyAPIError 直接抛出。）

        :param max_results: 最大结果数
        :return: 搜索结果列表
        """
        try:
            results = await self.client.search(
                self.query,
                search_depth="basic",
                max_results=max_results,
                topic=self.topic,
            )
            sources = results.get("results", [])
            if sources:
                return [
                    {"href": obj["url"], "body": obj["content"]} for obj in sources
                ]
            logger.info("Tavily API搜索未找到结果。回退到DuckDuckGo搜索API...")
        except Exception as e:
            logger.warning("Tavily搜索出错: %s。回退到DuckDuckGo搜索API...", e)

        # DuckDuckGo 回退（ddgs 仅提供同步接口，放入线程执行以免阻塞事件循环）
        try:
            results = await asyncio.to_thread(self._ddg_search, max_results)
            if not results:
                logger.info("DuckDuckGo搜索同样未找到结果。")
            return results
        except Exception as ddg_error:
            logger.error("DuckDuckGo搜索出错: %s。获取源失败。返回空响应。", ddg_error)
            return []

    def _ddg_search(self, max_results: int) -> List[Dict[str, str]]:
        """同步 DuckDuckGo 检索，仅供 :meth:`search` 经 ``asyncio.to_thread`` 调用。"""
        return list(DDGS().text(self.query, region="wt-wt", max_results=max_results))


def get_retriever(retriever: str):
    """
    按名称获取检索器类。

    :param retriever: 检索器名称
    :return: 对应的检索器类；未知名称返回 None（由调用方回退到默认检索器）
    """
    retrievers = {"tavily": TavilySearch}
    return retrievers.get(retriever)


def get_default_retriever():
    """
    获取默认检索器

    :return: 默认检索器类
    """
    return TavilySearch


class BeautifulSoupScraper:
    """使用BeautifulSoup的网页抓取器"""

    def __init__(self, link: str, config=None):
        """
        初始化抓取器

        :param link: 要抓取的URL
        :param config: 配置对象（可选，用于读取超时等设置）
        """
        self.link = link
        self.config = config

    async def scrape(self, client: httpx.AsyncClient) -> str:
        """
        抓取网页内容（异步）

        :param client: 复用的 httpx 异步客户端（由调用方创建并统一管理连接池）
        :return: 抓取的内容（失败或非 HTML 资源时返回空字符串）
        """
        try:
            response = await client.get(self.link)
            response.raise_for_status()  # 检查 HTTP 错误状态码

            # 跳过 PDF 等非 HTML 资源，避免二进制内容污染文本抽取
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type:
                return ""

            soup = BeautifulSoup(
                response.content, "lxml", from_encoding=response.encoding
            )

            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()

            raw_content = self._get_content_from_url(soup)
            lines = (line.strip() for line in raw_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return "\n".join(chunk for chunk in chunks if chunk)

        except httpx.HTTPError as e:
            logger.warning("网络请求错误 %s: %s", self.link, e)
            return ""
        except Exception as e:
            logger.warning("抓取 %s 时出错: %s", self.link, e)
            return ""

    def _get_content_from_url(self, soup: BeautifulSoup) -> str:
        """
        从BeautifulSoup对象中提取内容

        :param soup: BeautifulSoup对象
        :return: 提取的文本内容
        """
        text = ""
        tags = ["p", "h1", "h2", "h3", "h4", "h5"]
        for element in soup.find_all(tags):
            text += element.text + "\n"
        return text


class Scraper:
    """网页抓取器"""

    def __init__(
        self,
        urls: List[str],
        user_agent: str,
        config=None,
        scraper_class=BeautifulSoupScraper,
    ):
        """
        初始化抓取器

        :param urls: 要抓取的URL列表
        :param user_agent: 用户代理字符串
        :param config: 配置对象（可选，用于读取并发数、最小内容长度等设置）
        :param scraper_class: 使用的抓取器类
        """
        self.urls = urls
        self.user_agent = user_agent
        self.scraper_class = scraper_class
        self.config = config

    async def run(self) -> List[Dict[str, Any]]:
        """
        执行抓取任务（异步并发）

        复用单个 httpx 异步客户端以共享连接池，并用 ``asyncio.gather`` 并发抓取；
        并发上限由连接池 ``max_connections`` 控制（httpx 官方推荐做法）。

        :return: 抓取结果列表
        """
        max_connections = (
            getattr(self.config, "scrape_max_workers", 20) if self.config else 20
        )
        timeout = getattr(self.config, "scrape_timeout", 8) if self.config else 8
        async with httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections),
        ) as client:
            contents = await asyncio.gather(
                *(self._extract_data_from_link(client, link) for link in self.urls)
            )
        return [content for content in contents if content["raw_content"] is not None]

    async def _extract_data_from_link(
        self, client: httpx.AsyncClient, link: str
    ) -> Dict[str, Any]:
        """
        从链接中提取数据（异步）

        :param client: 复用的 httpx 异步客户端
        :param link: 要抓取的URL
        :return: 包含抓取结果的字典
        """
        try:
            scraper = self.scraper_class(link, self.config)
            content = await scraper.scrape(client)

            min_length = (
                getattr(self.config, "min_content_length", 100) if self.config else 100
            )
            if len(content) < min_length:
                return {"url": link, "raw_content": None}
            return {"url": link, "raw_content": content}
        except Exception as e:
            logger.warning("从 %s 提取数据时出错: %s", link, e)
            return {"url": link, "raw_content": None}


async def scrape_urls(urls: List[str], cfg: Any) -> List[Dict[str, Any]]:
    """
    抓取多个URL（异步）

    :param urls: 要抓取的URL列表
    :param cfg: 配置对象
    :return: 抓取结果列表
    """
    scraper = Scraper(urls, cfg.user_agent, config=cfg)
    return await scraper.run()


class SearchAPIRetriever(BaseRetriever):
    """搜索API检索器"""

    pages: List[Dict[str, Any]] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        """
        获取相关文档（langchain 检索器契约方法，经 ``.invoke`` 调用）。

        :param query: 查询字符串
        :param run_manager: 检索回调管理器（由 langchain 传入）
        :return: 相关文档列表
        """
        return [
            Document(
                page_content=page.get("raw_content", ""),
                metadata={
                    "title": page.get("title", ""),
                    "source": page.get("url", ""),
                },
            )
            for page in self.pages
        ]


class ContextCompressor:
    """上下文压缩器"""

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Any,
        max_results: int = 5,
        **kwargs,
    ):
        """
        初始化上下文压缩器

        :param documents: 文档列表
        :param embeddings: 嵌入模型
        :param max_results: 最大结果数
        :param kwargs: 其他参数
        """
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.38))

    def get_contextual_retriever(self) -> ContextualCompressionRetriever:
        """
        获取上下文压缩检索器

        :return: 上下文压缩检索器
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(
            embeddings=self.embeddings, similarity_threshold=self.similarity_threshold
        )
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever()
        base_retriever.pages = self.documents
        return ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )

    def pretty_print_docs(self, docs: List[Document], top_n: int) -> str:
        """
        美化打印文档

        :param docs: 文档列表
        :param top_n: 要打印的顶部文档数
        :return: 格式化的文档字符串
        """
        return "\n".join(
            f"来源: {d.metadata.get('source')}\n"
            f"标题: {d.metadata.get('title')}\n"
            f"内容: {d.page_content}\n"
            for i, d in enumerate(docs)
            if i < top_n
        )

    async def get_context(self, query: str, max_results: int = 5) -> str:
        """
        获取查询的上下文

        :param query: 查询字符串
        :param max_results: 最大结果数
        :return: 压缩后的上下文字符串
        """
        compressed_docs = self.get_contextual_retriever()
        relevant_docs = await compressed_docs.ainvoke(query)
        return self.pretty_print_docs(relevant_docs, max_results)
