"""
SQL助手图构建模块。
构建和配置SQL助手的完整处理流程图。
"""

import uuid
import logging
import os
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langfuse.callback import CallbackHandler

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from backend.sql_assistant.nodes.intent_analysis_node import intent_analysis_node
from backend.sql_assistant.nodes.keyword_extraction_node import keyword_extraction_node
from backend.sql_assistant.nodes.term_mapping_node import domain_term_mapping_node
from backend.sql_assistant.nodes.query_rewrite_node import query_rewrite_node
from backend.sql_assistant.nodes.data_source_node import data_source_identification_node
from backend.sql_assistant.nodes.table_structure_node import (
    table_structure_analysis_node,
)
from backend.sql_assistant.nodes.sql_generation_node import sql_generation_node
from backend.sql_assistant.nodes.permission_control_node import permission_control_node
from backend.sql_assistant.nodes.sql_execution_node import sql_execution_node
from backend.sql_assistant.nodes.error_analysis_node import error_analysis_node
from backend.sql_assistant.nodes.result_generation_node import result_generation_node
from backend.sql_assistant.nodes.feasibility_check_node import feasibility_check_node
from backend.sql_assistant.routes.node_routes import (
    route_after_intent,
    route_after_execution,
    route_after_error_analysis,
    route_after_feasibility_check,
    route_after_permission_check,
)

logger = logging.getLogger(__name__)


def create_langfuse_handler(session_id: str) -> CallbackHandler:
    """
    创建Langfuse回调处理器。

    Args:
        session_id: 会话ID

    Returns:
        CallbackHandler: Langfuse回调处理器实例
    """
    return CallbackHandler(tags=["sql_assistant"], session_id=session_id)


def build_sql_assistant_graph() -> StateGraph:
    """构建SQL助手的完整处理图

    创建并配置所有节点和边，设置路由逻辑。
    包括意图分析、关键词提取、业务术语规范化等全流程。

    Returns:
        StateGraph: 配置好的状态图实例
    """
    # 创建图构建器
    graph_builder = StateGraph(SQLAssistantState)

    # 添加所有节点
    graph_builder.add_node("intent_analysis", intent_analysis_node)
    graph_builder.add_node("keyword_extraction", keyword_extraction_node)
    graph_builder.add_node("domain_term_mapping", domain_term_mapping_node)
    graph_builder.add_node("query_rewrite", query_rewrite_node)
    graph_builder.add_node(
        "data_source_identification", data_source_identification_node
    )
    graph_builder.add_node("table_structure_analysis", table_structure_analysis_node)
    graph_builder.add_node("feasibility_checking", feasibility_check_node)
    graph_builder.add_node("sql_generation", sql_generation_node)
    # 添加权限控制节点
    graph_builder.add_node("permission_control", permission_control_node)
    graph_builder.add_node("sql_execution", sql_execution_node)
    graph_builder.add_node("error_analysis", error_analysis_node)
    graph_builder.add_node("result_generation", result_generation_node)

    # 设置条件边
    graph_builder.add_conditional_edges(
        "intent_analysis",
        route_after_intent,
        {"keyword_extraction": "keyword_extraction", END: END},
    )

    graph_builder.add_conditional_edges(
        "feasibility_checking",
        route_after_feasibility_check,
        {"sql_generation": "sql_generation", END: END},
    )

    graph_builder.add_conditional_edges(
            "permission_control",
            route_after_permission_check,
            {
                "sql_execution": "sql_execution",
                "error_analysis": "error_analysis"
            }
        )

    graph_builder.add_conditional_edges(
        "sql_execution",
        route_after_execution,
        {"result_generation": "result_generation", "error_analysis": "error_analysis"},
    )

    graph_builder.add_conditional_edges(
        "error_analysis",
        route_after_error_analysis,
        {"sql_execution": "sql_execution", END: END},
    )

    # 修改基本流程边,添加权限控制节点
    graph_builder.add_edge("keyword_extraction", "domain_term_mapping")
    graph_builder.add_edge("domain_term_mapping", "query_rewrite")
    graph_builder.add_edge("query_rewrite", "data_source_identification")
    graph_builder.add_edge("data_source_identification", "table_structure_analysis")
    graph_builder.add_edge("table_structure_analysis", "feasibility_checking")
    graph_builder.add_edge("sql_generation", "permission_control")
    graph_builder.add_edge("result_generation", END)

    # 设置入口
    graph_builder.add_edge(START, "intent_analysis")

    return graph_builder


def run_sql_assistant(
    query: str,
    thread_id: Optional[str] = None,
    checkpoint_saver: Optional[Any] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """运行SQL助手

    Args:
        query: 用户的查询文本
        thread_id: 会话ID
        checkpoint_saver: 状态保存器实例
        user_id: 用户ID,用于权限控制

    Returns:
        Dict[str, Any]: 处理结果字典
    """
    # 创建图构建器
    graph_builder = build_sql_assistant_graph()

    # 使用默认的内存存储器
    if checkpoint_saver is None:
        checkpoint_saver = MemorySaver()

    # 编译图
    graph = graph_builder.compile(checkpointer=checkpoint_saver)

    # 生成会话ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # 配置运行参数
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # 仅在启用 Langfuse 时添加 callbacks
    if os.getenv("LANGFUSE_ENABLED", "false").lower() == "true":
        config["callbacks"] = [create_langfuse_handler(thread_id)]

    # 构造输入状态
    state_input = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,  # 在初始状态中加入用户ID
    }

    # 执行图
    try:
        return graph.invoke(state_input, config)
    except Exception as e:
        error_msg = f"SQL助手执行出错: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
