"""
数据源识别节点模块。
负责识别与查询需求最相关的数据表。
"""

import os
import logging
from typing import List, Dict, Any

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from utils.vector_db_utils import (
    connect_to_milvus,
    initialize_vector_store,
    search_in_milvus
)
from utils.llm_tools import CustomEmbeddings

logger = logging.getLogger(__name__)


class DataSourceMatcher:
    """数据源匹配器

    负责识别与查询需求最相关的数据表。
    使用向量相似度匹配表的描述信息，支持模糊匹配。
    """

    def __init__(self):
        """初始化数据源匹配器"""
        # 连接到Milvus向量数据库
        connect_to_milvus(os.getenv("VECTOR_DB_DATABASE", ""))
        # 初始化表描述集合
        self.collection = initialize_vector_store("table_descriptions")
        # 初始化embedding模型
        self.embeddings = CustomEmbeddings(
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            api_url=os.getenv("EMBEDDING_API_BASE", ""),
            model=os.getenv("EMBEDDING_MODEL", "")
        )

    def find_relevant_tables(
        self,
        query: str,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """识别与查询最相关的数据表

        Args:
            query: 规范化后的查询文本
            top_k: 返回的最相关表数量

        Returns:
            List[Dict[str, Any]]: 相关表信息列表，每个表包含名称、描述和相似度分数

        Raises:
            ValueError: 向量搜索失败时抛出
        """
        try:
            # 生成查询文本的向量表示
            query_vector = self.embeddings.embed_query(query)

            # 在向量库中搜索相似表
            results = search_in_milvus(
                collection=self.collection,
                query_vector=query_vector,
                vector_field="description",
                top_k=top_k
            )

            # 转换结果格式
            return [
                {
                    "table_name": result["table_name"],
                    "description": result["description"],
                    "similarity_score": result["distance"],
                    "additional_info": result.get("additional_info", "")
                }
                for result in results
            ]

        except Exception as e:
            raise ValueError(f"数据表向量搜索失败: {str(e)}")


def data_source_identification_node(state: SQLAssistantState) -> dict:
    """数据源识别节点函数

    根据规范化后的查询需求，识别最相关的数据表。
    使用向量相似度匹配，支持模糊匹配和相关性排序。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含相关数据表信息的状态更新
    """
    # 获取改写后的查询
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        raise ValueError("状态中未找到改写后的查询")

    try:
        # 创建匹配器实例
        matcher = DataSourceMatcher()

        # 执行数据表匹配
        matched_tables = matcher.find_relevant_tables(rewritten_query)
        
        logger.info(f"匹配到的相关数据表: {[table['table_name'] for table in matched_tables]}")

        # 更新状态
        return {
            "matched_tables": matched_tables
        }

    except Exception as e:
        error_msg = f"数据源识别过程出错: {str(e)}"
        logger.error(error_msg)
        return {
            "matched_tables": [],
            "error": error_msg
        }