"""对话式数据核对使用的网络搜索工具。

- Tavily：官方 langchain 集成包 ``langchain-tavily`` 的 ``TavilySearch``；
- DuckDuckGo：维护中的 ``ddgs`` 库（``duckduckgo-search`` 已冻结并改名为 ``ddgs``）。
"""

import asyncio

from ddgs import DDGS
from langchain_tavily import TavilySearch


class SearchTools:
    """提供 DuckDuckGo 与 Tavily 两种网络搜索方法。"""

    def __init__(self, max_results: int = 5):
        """
        初始化搜索工具。

        Args:
            max_results (int): 每次搜索返回的最大结果数。
        """
        self.max_results = max_results
        self.tavily_search_tool = TavilySearch(max_results=max_results)

    async def aduckduckgo_search(self, query: str) -> str:
        """
        使用 DuckDuckGo（ddgs）执行搜索。

        Args:
            query (str): 搜索查询。

        Returns:
            str: 结果摘要文本列表的字符串表示。
        """

        def _search() -> list:
            return list(DDGS().text(query, max_results=self.max_results))

        results = await asyncio.to_thread(_search)
        snippets = [item.get("body", "") for item in results if item.get("body")]
        return str(snippets)

    async def atavily_search(self, query: str) -> str:
        """
        使用 Tavily 执行搜索并清洗结果。

        Args:
            query (str): 搜索查询。

        Returns:
            str: 清洗后内容摘要列表的字符串表示。
        """
        raw_results = await self.tavily_search_tool.ainvoke({"query": query})
        results = raw_results.get("results", []) if isinstance(raw_results, dict) else []
        cleaned_results = [item["content"] for item in results if "content" in item]
        return str(cleaned_results)
