from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults


class SearchTools:
    """
    搜索工具类，提供多种搜索方法。
    """

    def __init__(self):
        """
        初始化搜索工具。
        """
        self.duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        self.duckduckgo_search_tool = DuckDuckGoSearchRun(
            api_wrapper=self.duckduckgo_wrapper
        )
        self.tavily_search_tool = TavilySearchResults(max_results=5)

    async def aduckduckgo_search(self, query: str) -> str:
        """
        使用 DuckDuckGo 执行搜索。

        Args:
            query (str): 搜索查询。

        Returns:
            str: 搜索结果。
        """
        return await self.duckduckgo_search_tool.ainvoke(query)

    async def atavily_search(self, query: str) -> str:
        """
        使用 Tavily 执行搜索并清洗结果。

        Args:
            query (str): 搜索查询。

        Returns:
            list[str]: 清洗后的搜索结果列表，每个元素为一条搜索结果的内容。
        """
        raw_results = await self.tavily_search_tool.ainvoke(query)
        cleaned_results = [item["content"] for item in raw_results if "content" in item]
        return str(cleaned_results)
