# ai_research_config.py

import os
from typing import Optional


class Config:
    """AI研究助手配置类"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置

        :param config_file: 可选的配置文件路径
        :raises EnvironmentError: 如果必要的环境变量未设置
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "必要的环境变量未设置。请确保 .env 文件存在并包含所需的变量。"
            )

        self.retriever = os.getenv("RETRIEVER", "tavily")
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.38))
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.llm_model = "gpt-4-1106-preview"
        self.fast_token_limit = int(os.getenv("FAST_TOKEN_LIMIT", 4000))
        self.smart_token_limit = int(os.getenv("SMART_TOKEN_LIMIT", 4000))
        self.browse_chunk_max_length = int(os.getenv("BROWSE_CHUNK_MAX_LENGTH", 8192))
        self.summary_token_limit = int(os.getenv("SUMMARY_TOKEN_LIMIT", 700))
        self.temperature = float(os.getenv("TEMPERATURE", 0.55))
        self.user_agent = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        )
        self.max_search_results_per_query = int(
            os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", 5)
        )
        self.memory_backend = os.getenv("MEMORY_BACKEND", "local")
        self.total_words = int(os.getenv("TOTAL_WORDS", 800))
        self.report_format = os.getenv("REPORT_FORMAT", "APA")
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 5))
        self.agent_role = os.getenv("AGENT_ROLE")
        self.scraper = os.getenv("SCRAPER", "bs")
        self.max_subtopics = int(os.getenv("MAX_SUBTOPICS", 5))
        self.doc_path = os.getenv("DOC_PATH", "")
        self.llm_kwargs = {}

        if self.doc_path:
            self.validate_doc_path()

    def validate_doc_path(self):
        """验证并创建文档路径"""
        os.makedirs(self.doc_path, exist_ok=True)
