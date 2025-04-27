"""
业务术语映射节点模块。
负责将用户输入的非标准术语映射到标准的领域术语定义。
"""

import os
import logging
from typing import Dict, List

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from backend.sql_assistant.utils.format_utils import format_term_descriptions
from utils.vector_db_utils import (
    connect_to_milvus,
    initialize_vector_store,
    search_in_milvus,
)
from utils.llm_tools import CustomEmbeddings

logger = logging.getLogger(__name__)


class DomainTermMapper:
    """业务领域术语映射器

    负责将用户输入的非标准术语映射到标准的领域术语定义。
    使用向量数据库进行语义相似度匹配，支持模糊匹配和同义词处理。
    """

    def __init__(self):
        """初始化术语映射器"""
        # 连接到Milvus向量数据库
        connect_to_milvus(os.getenv("VECTOR_DB_DATABASE", ""))
        # 初始化术语描述集合
        self.collection = initialize_vector_store("term_descriptions")
        # 初始化embedding模型
        self.embeddings = CustomEmbeddings(
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            api_url=os.getenv("EMBEDDING_API_BASE", ""),
            model=os.getenv("EMBEDDING_MODEL", ""),
        )

    def find_standard_terms(
        self, keywords: List[str], similarity_threshold: float = 0.9
    ) -> Dict[str, Dict[str, str]]:
        """查找关键词对应的标准术语及其信息

        Args:
            keywords: 需要标准化的关键词列表
            similarity_threshold: 相似度匹配阈值，控制匹配的严格程度

        Returns:
            Dict[str, Dict[str, str]]: 关键词到标准术语信息的映射字典
        """
        if not keywords:
            return {}

        term_mappings = {}

        for keyword in keywords:
            try:
                query_vector = self.embeddings.embed_query(keyword)

                results = search_in_milvus(
                    collection=self.collection,
                    query_vector=query_vector,
                    vector_field="original_term",
                    top_k=1,
                )

                if results and results[0]["distance"] > similarity_threshold:
                    term_mappings[keyword] = {
                        "original_term": results[0]["original_term"],
                        "standard_name": results[0]["standard_name"],
                        "additional_info": results[0]["additional_info"],
                    }

            except Exception as e:
                logger.error(f"处理关键词 '{keyword}' 时发生错误: {str(e)}")
                continue

        return term_mappings


def domain_term_mapping_node(state: SQLAssistantState) -> dict:
    """领域术语映射节点函数

    将提取的关键词映射到标准领域术语，并获取其规范定义。
    这个步骤确保后续处理使用统一的业务术语。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含标准化术语及其解释的状态更新
    """
    # 获取关键词列表
    keywords = state.get("keywords", [])

    try:
        # 创建标准化处理器实例
        standardizer = DomainTermMapper()

        # 执行术语标准化
        term_mappings = standardizer.find_standard_terms(keywords)

        logger.info(f"术语映射结果: {term_mappings}")

        # 更新状态
        return {"domain_term_mappings": term_mappings}

    except Exception as e:
        error_msg = f"业务术语规范化过程出错: {str(e)}"
        logger.error(error_msg)
        return {"domain_term_mappings": {}, "error": error_msg}
