"""
数据源识别节点模块。
负责识别与查询需求最相关的数据表。

采用两阶段策略：先用向量相似度宽召回候选表，再由大模型从候选中
精选最相关的数据表，并给出相关性门控标志（has_relevant_tables）。
当没有任何相关数据表时，直接结束流程并给出友好提示，避免对不相关
的表强行生成 SQL 而产出误导性结果。
"""

import os
import logging
from typing import List, Dict, Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from utils.vector_db_utils import (
    connect_to_milvus,
    initialize_vector_store,
    search_in_milvus,
)
from utils.llm_tools import CustomEmbeddings, init_language_model, LanguageModelChain

logger = logging.getLogger(__name__)


class TableSelectionResult(BaseModel):
    """数据表选择结果模型"""

    selection_reasoning: str = Field(
        ..., description="选择数据表的简要推理过程"
    )
    selected_table_names: List[str] = Field(
        ..., description="从候选表中选出的最相关表名（最多3张表）"
    )


TABLE_SELECTION_SYSTEM_PROMPT = """你是一位专业的数据分析师，负责从候选数据表中选出最适合回答用户查询的数据表。
请遵循以下原则：

1. 理解查询需求：
   - 分析用户想要查询的核心信息
   - 识别查询的主要维度和指标
   - 判断是否需要多表关联

2. 评估表的相关性：
   - 表的主要用途是否与查询目标直接相关
   - 表中是否包含所需的关键字段
   - 表的数据粒度是否合适
   - 表的数据更新特性是否满足需求

3. 选择原则：
   - 根据查询需求选择最相关的表
   - 最多选择三张表
   - 选择数据粒度最合适的表
   - 避免选择冗余的表
   - 如果候选表中没有任何一张与查询需求真正相关，则返回空列表，不要勉强选择

4. 输出要求：
   - 用简短的话说明推理过程和选择理由（20字以内）
"""


TABLE_SELECTION_USER_PROMPT = """请从以下候选数据表中，选出最适合回答用户查询的数据表：

1. 用户查询需求：
{query}

2. 候选数据表列表：
{candidate_tables}

请分析每张表的相关性，选出最多2张最合适的表；如果没有任何一张真正相关，请返回空列表。"""


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
            model=os.getenv("EMBEDDING_MODEL", ""),
        )

    def find_candidate_tables(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """识别与查询相关的候选数据表（宽召回）

        Args:
            query: 规范化后的查询文本
            top_k: 返回的候选表数量

        Returns:
            List[Dict[str, Any]]: 候选表信息列表，每个表包含名称、描述和相似度分数

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
                top_k=top_k,
            )

            # 转换结果格式
            return [
                {
                    "table_name": result["table_name"],
                    "description": result["description"],
                    "similarity_score": result["distance"],
                    "additional_info": result.get("additional_info", ""),
                }
                for result in results
            ]

        except Exception as e:
            raise ValueError(f"数据表向量搜索失败: {str(e)}")


def create_table_selection_chain(temperature: float = 0.0) -> LanguageModelChain:
    """创建数据表选择任务链"""
    llm = init_language_model(temperature=temperature)

    return LanguageModelChain(
        model_cls=TableSelectionResult,
        sys_msg=TABLE_SELECTION_SYSTEM_PROMPT,
        user_msg=TABLE_SELECTION_USER_PROMPT,
        model=llm,
    )()


def format_candidate_tables(tables: List[Dict[str, Any]]) -> str:
    """格式化候选表信息，供大模型选择"""
    formatted = []
    for i, table in enumerate(tables, 1):
        formatted.append(
            f"{i}. 表名: {table['table_name']}\n"
            f"   描述: {table['description']}\n"
            f"   补充信息: {table['additional_info']}\n"
        )
    return "\n".join(formatted)


def data_source_identification_node(state: SQLAssistantState) -> dict:
    """数据源识别节点函数

    先基于向量相似度宽召回候选数据表，再由大模型从候选中精选最相关的表。
    若没有任何相关的表，则置门控标志并给出友好提示，交由路由结束流程。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含相关数据表信息与相关性门控标志的状态更新
    """
    # 获取改写后的查询
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        raise ValueError("状态中未找到改写后的查询")

    try:
        # 创建匹配器实例
        matcher = DataSourceMatcher()

        # 1. 向量宽召回候选表
        candidate_tables = matcher.find_candidate_tables(rewritten_query)
        logger.info(f"初步召回的候选数据表数量: {len(candidate_tables)}")

        # 2. 大模型从候选中精选最相关的表
        selection_chain = create_table_selection_chain()
        selection_result = selection_chain.invoke(
            {
                "query": rewritten_query,
                "candidate_tables": format_candidate_tables(candidate_tables),
            }
        )
        selected_table_names = selection_result["selected_table_names"]
        has_relevant_tables = len(selected_table_names) > 0

        logger.info(
            f"大模型选表结果: {selected_table_names}；"
            f"选择理由: {selection_result['selection_reasoning']}"
        )

        # 3. 从候选表中取出被选中表的完整信息
        matched_tables = [
            table
            for table in candidate_tables
            if table["table_name"] in selected_table_names
        ]

        # 4. 组装状态更新；若无相关表，追加友好提示消息
        response = {
            "matched_tables": matched_tables,
            "has_relevant_tables": has_relevant_tables,
        }

        if not has_relevant_tables:
            response["messages"] = [
                AIMessage(
                    content="抱歉，系统中没有找到与您查询相关的数据表。\n"
                    "建议：\n"
                    "- 尝试换一种说法或使用不同的关键词描述您的需求\n"
                    "- 联系数据团队确认是否有相应的数据支持"
                )
            ]

        return response

    except Exception as e:
        error_msg = f"数据源识别过程出错: {str(e)}"
        logger.error(error_msg)
        return {
            "matched_tables": [],
            "has_relevant_tables": False,
            "error": error_msg,
        }
