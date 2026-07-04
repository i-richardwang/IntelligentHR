"""SQL 助手路由函数单元测试。

回归保护 wave5 的两个增强：
- A3 选表相关性门控：data_source 识别后若无相关表则直接结束（END），否则进入表结构分析。
- B1 少样本：可行性检查通过后进入查询示例检索（query_example_retrieval），而非直接生成 SQL。
以及原有的意图 / 执行 / 错误分析 / 权限路由逻辑。
"""

from langgraph.graph import END

from backend.sql_assistant.routes.node_routes import (
    route_after_intent,
    route_after_data_source,
    route_after_feasibility_check,
    route_after_execution,
    route_after_error_analysis,
    route_after_permission_check,
)


def test_intent_clear_continues():
    assert route_after_intent({"is_intent_clear": True}) == "keyword_extraction"


def test_intent_unclear_ends():
    assert route_after_intent({"is_intent_clear": False}) == END


# ---------------- A3：选表相关性门控 ----------------


def test_data_source_with_relevant_tables():
    assert (
        route_after_data_source({"has_relevant_tables": True})
        == "table_structure_analysis"
    )


def test_data_source_without_relevant_tables_ends():
    assert route_after_data_source({"has_relevant_tables": False}) == END


def test_data_source_missing_flag_defaults_to_end():
    # 缺失门控标志时应保守结束，避免带空表继续下游
    assert route_after_data_source({}) == END


# ---------------- B1：少样本检索接入 ----------------


def test_feasibility_feasible_goes_to_examples():
    state = {"feasibility_check": {"is_feasible": True}}
    assert route_after_feasibility_check(state) == "query_example_retrieval"


def test_feasibility_infeasible_ends():
    state = {"feasibility_check": {"is_feasible": False}}
    assert route_after_feasibility_check(state) == END


def test_feasibility_missing_ends():
    assert route_after_feasibility_check({}) == END


# ---------------- 执行 / 错误分析 / 权限 ----------------


def test_execution_success():
    assert (
        route_after_execution({"execution_result": {"success": True}})
        == "result_generation"
    )


def test_execution_failure():
    assert (
        route_after_execution({"execution_result": {"success": False}})
        == "error_analysis"
    )


def test_error_analysis_fixable_reexecutes():
    state = {"error_analysis_result": {"is_sql_fixable": True, "fixed_sql": "SELECT 1"}}
    assert route_after_error_analysis(state) == "sql_execution"
    # 应把修复后的 SQL 写回状态，供重试执行
    assert state["generated_sql"] == {"sql_query": "SELECT 1"}


def test_error_analysis_unfixable_ends():
    state = {"error_analysis_result": {"is_sql_fixable": False}}
    assert route_after_error_analysis(state) == END


def test_permission_pass_executes():
    state = {"execution_result": {"success": True}}
    assert route_after_permission_check(state) == "sql_execution"


def test_permission_fail_analyzes():
    state = {"execution_result": {"success": False}}
    assert route_after_permission_check(state) == "error_analysis"
