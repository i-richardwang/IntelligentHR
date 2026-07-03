"""
查询示例检索节点模块。
负责根据用户查询检索最相似的 SQL 查询示例，作为少样本（few-shot）
参考注入到 SQL 生成环节，提升生成 SQL 的准确性与一致性。

依赖 Milvus 中的 query_examples 集合（字段包含 query_text、query_sql）。
若该集合尚未创建或检索失败，节点将安全返回空列表，不影响主流程。
"""

import os
import logging
from typing import Dict, List

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from utils.vector_db_utils import (
    connect_to_milvus,
    initialize_vector_store,
    search_in_milvus,
)
from utils.llm_tools import CustomEmbeddings

logger = logging.getLogger(__name__)


class QueryExampleRetriever:
    """查询示例检索器

    从 Milvus 的 query_examples 集合中检索与用户查询最相似的
    “自然语言问题 -> SQL” 样例对。
    """

    def __init__(self):
        """初始化查询示例检索器"""
        # 连接到Milvus向量数据库
        connect_to_milvus(os.getenv("VECTOR_DB_DATABASE", ""))
        # 初始化查询示例集合（集合不存在时会抛出 ValueError）
        self.collection = initialize_vector_store("query_examples")
        # 初始化embedding模型
        self.embeddings = CustomEmbeddings(
            api_key=os.getenv("EMBEDDING_API_KEY", ""),
            api_url=os.getenv("EMBEDDING_API_BASE", ""),
            model=os.getenv("EMBEDDING_MODEL", ""),
        )

    def retrieve_examples(
        self, query: str, top_k: int = 2
    ) -> List[Dict[str, str]]:
        """根据查询检索最相似的 SQL 查询示例

        Args:
            query: 用户查询文本
            top_k: 返回的最相似示例数量

        Returns:
            List[Dict[str, str]]: 最相似的查询示例列表
        """
        # 生成查询文本的向量表示
        query_vector = self.embeddings.embed_query(query)

        # 在向量库中搜索相似示例
        results = search_in_milvus(
            collection=self.collection,
            query_vector=query_vector,
            vector_field="query_text",
            top_k=top_k,
        )

        # 转换结果格式
        return [
            {
                "query_text": result["query_text"],
                "query_sql": result["query_sql"],
            }
            for result in results
        ]


def query_example_node(state: SQLAssistantState) -> dict:
    """查询示例检索节点函数

    从向量数据库中检索与当前查询最相似的 SQL 查询示例。
    该节点为增强性节点：即便 query_examples 集合不存在或检索失败，
    也会安全返回空列表，保证主流程正常进行。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含检索到的查询示例的状态更新
    """
    # 获取改写后的查询
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        return {"query_examples": []}

    try:
        # 创建示例检索器实例
        retriever = QueryExampleRetriever()

        # 检索相似示例
        query_examples = retriever.retrieve_examples(rewritten_query)

        logger.info(f"检索到 {len(query_examples)} 条相似查询示例")

        return {"query_examples": query_examples}

    except Exception as e:
        # 集合不存在或检索失败时安全降级，不阻断主流程
        logger.warning(f"查询示例检索跳过（不影响主流程）: {str(e)}")
        return {"query_examples": []}
