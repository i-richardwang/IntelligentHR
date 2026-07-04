# backend/ai_research/ai_researcher.py

import asyncio
import logging
from typing import List, Dict, Any, Optional

from backend.ai_research.ai_research_config import Config
from backend.ai_research.research_enums import ReportType, ReportSource, Tone
from backend.ai_research.web_retriever import (
    get_retriever,
    get_default_retriever,
    scrape_urls,
    ContextCompressor,
)
from backend.ai_research.embedding_service import Memory
from backend.ai_research.ai_research_agent import (
    choose_agent,
    get_sub_queries,
    construct_subtopics,
    generate_report,
    get_report_introduction,
)

logger = logging.getLogger(__name__)


class AIResearcher:
    """AI 研究助手类"""

    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        report_source: str = ReportSource.Web.value,
        tone: Tone = Tone.Objective,
        config_path: Optional[str] = None,
        websocket: Any = None,
        agent: Optional[str] = None,
        role: Optional[str] = None,
        verbose: bool = True,
        verbose_callback: Optional[callable] = None,
        max_iterations: Optional[int] = None,
        max_subtopics: Optional[int] = None,
        max_search_results_per_query: Optional[int] = None,
    ):
        """
        初始化 AI 研究助手

        :param query: 研究查询
        :param report_type: 报告类型
        :param report_source: 报告来源
        :param tone: 报告语气
        :param config_path: 配置文件路径（可选）
        :param websocket: WebSocket 对象（可选）
        :param agent: 指定的代理（可选）
        :param role: 指定的角色（可选）
        :param verbose: 是否详细输出
        :param verbose_callback: 详细输出回调函数（可选）
        :param max_iterations: 最大迭代次数（可选）
        :param max_subtopics: 最大子主题数（可选）
        :param max_search_results_per_query: 每个查询的最大搜索结果数（可选）
        """
        if not query or not query.strip():
            raise ValueError("研究查询内容不能为空")
        self.query = query.strip()
        self.report_type = report_type
        self.report_source = report_source
        self.tone = tone
        self.cfg = Config(config_path)
        self.websocket = websocket
        self.agent = agent
        self.role = role
        self.verbose = verbose
        self.verbose_callback = verbose_callback
        self.context: List[str] = []
        self.memory = Memory(self.cfg.embedding_provider)

        # 更新配置
        if max_iterations is not None:
            self.cfg.max_iterations = max_iterations
        if max_subtopics is not None:
            self.cfg.max_subtopics = max_subtopics
        if max_search_results_per_query is not None:
            self.cfg.max_search_results_per_query = max_search_results_per_query

    def log(self, message: str) -> None:
        """
        记录日志信息

        :param message: 要记录的消息
        """
        if self.verbose:
            logger.info(message)
            if self.verbose_callback:
                self.verbose_callback(message)

    async def process_sub_query(self, sub_query: str, index: int, total: int) -> str:
        """
        处理子查询

        :param sub_query: 子查询
        :param index: 当前子查询索引
        :param total: 总子查询数
        :return: 压缩后的上下文
        """
        self.log(f"正在处理子查询...")

        retriever_class = get_retriever(self.cfg.retriever) or get_default_retriever()
        retriever = retriever_class(sub_query)
        self.log(f"使用检索器: {retriever.__class__.__name__}")

        search_results = await retriever.search(
            max_results=self.cfg.max_search_results_per_query
        )

        urls = [result["href"] for result in search_results]
        self.log(f"正在为子查询抓取 URL...")

        scraped_content = await scrape_urls(urls, self.cfg)

        context_compressor = ContextCompressor(
            scraped_content, self.memory.get_embeddings()
        )
        self.log(f"正在为子查询压缩上下文...")

        compressed_context = await context_compressor.get_context(sub_query)
        self.log(f"子查询的上下文压缩完成。")

        return compressed_context

    async def conduct_research(self) -> List[str]:
        """
        执行研究任务

        :return: 研究上下文列表
        """
        self.log(f"🔎 开始 '{self.query}' 的研究任务...")

        if not (self.agent and self.role):
            self.agent, self.role = await choose_agent(self.query, self.cfg)
        self.log(f"选择的代理: {self.agent}")

        sub_queries = await get_sub_queries(
            self.query, self.role, self.cfg, None, self.report_type
        )
        self.log(f"生成的子查询: {sub_queries}")

        self.log(f"开始为 {len(sub_queries)} 个子查询进行搜索和抓取...")

        semaphore = asyncio.Semaphore(5)  # 限制并发数量

        async def limited_process_sub_query(*args):
            async with semaphore:
                return await self.process_sub_query(*args)

        tasks = [
            limited_process_sub_query(sub_query, i + 1, len(sub_queries))
            for i, sub_query in enumerate(sub_queries)
        ]
        self.context = await asyncio.gather(*tasks)

        self.log(f"研究阶段完成。共收集上下文数量: {len(self.context)}")

        return self.context

    async def generate_report(self) -> str:
        """
        生成研究报告

        :return: 生成的报告内容
        """
        self.log("开始生成报告...")

        full_context = "\n".join(self.context)
        self.log(f"合并后的上下文长度: {len(full_context)} 字符")

        if self.report_type == ReportType.DetailedReport.value:
            self.log("生成详细报告...")

            self.log("构建子主题...")
            subtopics = await construct_subtopics(self.query, full_context, self.cfg)
            self.log(f"生成了 {len(subtopics)} 个子主题")

            self.log("生成报告引言...")
            introduction = await get_report_introduction(
                self.query, full_context, self.role, self.cfg
            )
            self.log("引言生成成功")

            subtopic_reports = []
            for i, subtopic in enumerate(subtopics, 1):
                self.log(
                    f"正在为子主题 {i}/{len(subtopics)} 生成报告: '{subtopic['task']}'"
                )
                subtopic_report = await generate_report(
                    subtopic["task"],
                    full_context,
                    self.role,
                    "subtopic_report",
                    self.tone,
                    self.report_source,
                    self.cfg,
                    main_topic=self.query,
                    existing_headers=[s["task"] for s in subtopics],
                )
                subtopic_reports.append(subtopic_report)
                self.log(f"子主题 {i} 的报告生成成功")

            full_report = f"{introduction}\n\n" + "\n\n".join(subtopic_reports)
            self.log("详细报告编译完成")
        else:
            self.log(f"生成 {self.report_type} 报告...")
            full_report = await generate_report(
                self.query,
                full_context,
                self.role,
                self.report_type,
                self.tone,
                self.report_source,
                self.cfg,
            )
            self.log("报告生成完成")

        self.log(f"最终报告长度: {len(full_report)} 字符")
        return full_report

    async def run(self) -> str:
        """
        运行 AI 研究助手

        :return: 生成的研究报告
        """
        await self.conduct_research()
        report = await self.generate_report()
        return report
