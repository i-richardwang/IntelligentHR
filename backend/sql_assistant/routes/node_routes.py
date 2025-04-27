"""
节点路由模块。
定义了状态图中各节点间的路由逻辑。
"""

from langgraph.graph import END

from backend.sql_assistant.states.assistant_state import SQLAssistantState


def route_after_intent(state: SQLAssistantState):
    """意图分析后的路由函数

    根据意图分析结果决定下一个处理节点。
    如果意图不明确，结束对话要求澄清；否则继续处理。

    Args:
        state: 当前状态对象

    Returns:
        str: 下一个节点的标识符
    """
    if not state["is_intent_clear"]:
        return END
    return "keyword_extraction"


def route_after_execution(state: SQLAssistantState):
    """SQL执行后的路由函数

    根据SQL执行结果决定下一步操作。
    执行成功则进入结果反馈，失败则进入错误分析。

    Args:
        state: 当前状态对象

    Returns:
        str: 下一个节点的标识符
    """
    execution_result = state.get("execution_result", {})
    if execution_result.get("success", False):
        return "result_generation"  # 执行成功，生成结果反馈
    return "error_analysis"  # 执行失败，进入错误分析


def route_after_error_analysis(state: SQLAssistantState):
    """错误分析后的路由函数

    根据错误分析结果决定下一步操作。
    如果错误可修复，则使用修复后的SQL重新执行；
    否则结束处理流程。

    Args:
        state: 当前状态对象

    Returns:
        str: 下一个节点的标识符
    """
    error_analysis_result = state.get("error_analysis_result", {})
    error_analysis_result = state.get("error_analysis_result", {})
    if error_analysis_result.get("is_sql_fixable", False):
        state["generated_sql"] = {"sql_query": error_analysis_result["fixed_sql"]}
        return "sql_execution"
    return END


def route_after_feasibility_check(state: SQLAssistantState):
    """可行性检查后的路由函数

    根据可行性检查结果决定下一步操作。
    如果查询可行则生成SQL，否则结束处理。

    Args:
        state: 当前状态对象

    Returns:
        str: 下一个节点的标识符
    """
    feasibility_check = state.get("feasibility_check", {})
    if not feasibility_check or not feasibility_check.get("is_feasible"):
        return END
    return "sql_generation"


def route_after_permission_check(state: SQLAssistantState):
    """权限检查后的路由函数

    根据执行结果决定下一个处理节点:
    - 验证通过: 进入SQL执行节点
    - 验证失败: 直接进入错误分析节点

    Args:
        state: 当前状态对象

    Returns:
        str: 下一个节点的标识符
    """
    execution_result = state.get("execution_result", {})

    if execution_result.get("success", False):
        return "sql_execution"
    return "error_analysis"
