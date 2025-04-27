"""
格式化工具函数模块。
提供各种数据格式化的通用函数。
"""

from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from tabulate import tabulate
import json


def format_conversation_history(messages: List[BaseMessage]) -> str:
    """将对话历史格式化为提示词格式

    将消息列表转换为结构化的对话历史文本，用于LLM输入

    Args:
        messages: 消息历史列表

    Returns:
        str: 格式化后的对话历史
    """
    formatted = []
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


def format_term_descriptions(term_mappings: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    """格式化业务术语映射信息

    Args:
        term_mappings: 业务术语映射信息字典

    Returns:
        List[Dict[str, str]]: 格式化后的术语映射信息列表
    """
    if not term_mappings:
        return []

    formatted_mappings = []
    for mapping in term_mappings.values():
        formatted_mappings.append({
            "original_term": mapping["original_term"],
            "standard_name": mapping["standard_name"],
            "additional_info": mapping["additional_info"]
        })
    
    return formatted_mappings


def format_table_structures(schemas: List[Dict]) -> str:
    """格式化表结构信息

    Args:
        schemas: 表结构信息列表

    Returns:
        str: 格式化后的表结构文本
    """
    formatted = []
    for schema in schemas:
        formatted.append(f"表名: {schema['table_name']}")
        formatted.append(f"描述: {schema.get('description', '暂无描述')}")
        if schema.get('additional_info'):
            formatted.append(f"使用说明: {schema['additional_info']}")
        formatted.append("字段列表:")
        formatted.append("| 字段名 | 类型 | 说明 |")
        formatted.append("|--------|------|------|")
        for col in schema['columns']:
            formatted.append(
                f"| {col['name']} | {col['type']} | {col['comment']} |"
            )
        formatted.append("")  # 添加空行分隔
    return "\n".join(formatted)


def format_results_preview(execution_result: Dict) -> str:
    """格式化查询结果预览

    将查询结果格式化为易读的表格形式。
    当结果超过20行时，不展示具体数据，而是返回提示信息。

    Args:
        execution_result: SQL执行结果字典

    Returns:
        str: 格式化后的结果预览文本
    """
    if not execution_result.get('results'):
        return "无数据"

    results = execution_result['results']
    columns = execution_result['columns']

    # 如果结果集过大，返回提示信息
    if len(results) > 20:
        return f"结果集过大，不展示具体数据"

    # 构建表格形式的预览
    lines = []
    # 表头
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["-" * len(col) for col in columns]) + "|")
    # 数据行
    for row in results:
        lines.append("| " + " | ".join(str(row[col])
                                       for col in columns) + " |")

    return "\n".join(lines)


def format_full_results(execution_result: Dict) -> str:
    """格式化完整的查询结果
    
    将查询结果格式化为完整的表格形式，使用 tabulate 生成美观的表格。
    
    Args:
        execution_result: SQL执行结果字典
        
    Returns:
        str: 格式化后的完整结果文本
    """
    if not execution_result.get('results'):
        return "无数据"
        
    results = execution_result['results']
    columns = execution_result['columns']
    
    # 将结果转换为行数据列表
    rows = []
    for row in results:
        rows.append([str(row[col]) for col in columns])
    
    # 使用 tabulate 生成表格
    # 使用 'pipe' 样式，这样在 notebook 中显示效果最好
    table = tabulate(
        rows,
        headers=columns,
        tablefmt='pipe',  # 使用 pipe 格式，在 notebook 中显示为 markdown 表格
        showindex=False,
        numalign='left',
        stralign='left'
    )
    
    return table
