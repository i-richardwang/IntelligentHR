"""
SQL助手状态定义模块。
定义了SQL助手在处理过程中的状态数据结构。
"""

from typing import Annotated, Dict, Optional, List, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

# 引入消息添加注解
from langgraph.graph.message import add_messages


class SQLAssistantState(TypedDict):
    """SQL助手的状态类型定义"""
    # 用户ID
    user_id: Optional[int]
    # 消息历史记录
    messages: Annotated[List[BaseMessage], add_messages]
    # 查询意图分析结果
    query_intent: Optional[Dict]
    # 意图是否明确的标志
    is_intent_clear: bool
    # 提取的关键词列表
    keywords: List[str]
    # 业务术语及其描述
    domain_term_mappings: Dict[str, str]
    # 改写后的查询
    rewritten_query: Optional[str]
    # 匹配的数据表信息
    matched_tables: List[Dict[str, Any]]
    # 数据表结构信息
    table_structures: List[Dict[str, Any]]
    # 生成的SQL信息
    generated_sql: Optional[Dict[str, Any]]
    # SQL执行结果
    execution_result: Optional[Dict[str, Any]]
    # SQL错误分析结果
    error_analysis_result: Optional[Dict[str, Any]]
    # SQL执行重试计数
    retry_count: int
    # 查询结果反馈
    result_feedback: Optional[str]
    # 可行性检查结果
    feasibility_check: Optional[Dict]