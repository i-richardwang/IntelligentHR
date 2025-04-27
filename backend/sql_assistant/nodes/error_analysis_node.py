"""
错误分析节点模块。
负责分析SQL执行错误的原因并提供修复方案。
"""

import logging
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import AIMessage

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from backend.sql_assistant.utils.format_utils import (
    format_table_structures,
    format_term_descriptions,
)
from utils.llm_tools import init_language_model, LanguageModelChain

logger = logging.getLogger(__name__)


class ErrorAnalysisResult(BaseModel):
    """SQL错误分析结果模型"""

    error_analysis: str = Field(..., description="错误原因思考和分析")
    is_sql_fixable: bool = Field(
        ..., description="判断是否是可以通过修改SQL来解决的错误"
    )
    fixed_sql: Optional[str] = Field(
        None, description="当错误可修复时，提供修正后的SQL语句"
    )
    user_feedback: Optional[str] = Field(None, description="面向用户的友好错误说明")


ERROR_ANALYSIS_SYSTEM_PROMPT = """你是一个专业的数据分析师，负责分析SQL执行失败的原因并提供解决方案。
请遵循以下规则进行分析：

1. 错误类型分析：
   - 语法错误：SQL语句本身的语法问题
   - 字段错误：表或字段名称错误、类型不匹配
   - 数据错误：空值处理、数据范围、类型转换等问题
   - 权限错误：数据库访问权限问题
   - 连接错误：数据库连接、网络等问题
   - 其他系统错误：数据库服务器问题等

2. 判断可修复性：
   - 可修复：通过修改SQL语句可以解决的问题（如语法错误、字段错误等）
   - 不可修复：需要系统层面解决的问题（如权限、连接等问题）

3. 修复建议：
   - 如果是可修复的错误，提供具体的修复后的SQL语句
   - 确保修复后的SQL符合原始查询意图
   - 保持SQL的优化原则
   - 确保使用正确的表和字段名

4. 用户反馈要求：
   - 对于权限问题：
     * 指出用户缺少哪些表的访问权限
     * 用自然语言表示表名，不要泄露数据库的表名
     * 建议用户联系数据团队申请相应权限
   - 对于不可修复的其他问题：
     * 用通俗易懂的语言解释问题
     * 提供明确的后续处理建议
     * 适当表达歉意

5. 输出要求：
   - 明确指出是否可以通过修改SQL修复
   - 详细解释错误原因
   - 对于可修复错误，提供修正后的SQL
   - 对于不可修复错误，提供用户友好的反馈信息"""

ERROR_ANALYSIS_USER_PROMPT = """请分析以下SQL执行失败的原因并提供解决方案：

1. 原始查询需求：
{rewritten_query}

2. 可用的表结构：
{table_structures}

3. 业务术语说明：
{term_descriptions}

4. 执行失败的SQL：
{failed_sql}

5. 错误信息：
{error_message}

请分析错误原因，判断是否可以通过修改SQL修复，并按照指定的JSON格式输出分析结果。
"""


def create_error_analysis_chain(temperature: float = 0.0) -> LanguageModelChain:
    """创建错误分析任务链"""
    llm = init_language_model(temperature=temperature)

    return LanguageModelChain(
        model_cls=ErrorAnalysisResult,
        sys_msg=ERROR_ANALYSIS_SYSTEM_PROMPT,
        user_msg=ERROR_ANALYSIS_USER_PROMPT,
        model=llm,
    )()


def error_analysis_node(state: SQLAssistantState) -> dict:
    """SQL错误分析节点函数

    分析SQL执行失败的原因，判断是否可修复，
    并在可能的情况下提供修复方案。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含错误分析结果的状态更新
    """
    # 获取执行结果
    execution_result = state.get("execution_result", {})
    if not execution_result or execution_result.get("success", True):
        return {"error": "状态中未找到失败的执行结果"}

    generated_sql = state.get("generated_sql", {})
    if not generated_sql or not generated_sql.get("sql_query"):
        return {"error": "状态中未找到生成的SQL"}

    try:
        # 准备输入数据
        input_data = {
            "rewritten_query": state["rewritten_query"],
            "table_structures": format_table_structures(state["table_structures"]),
            "term_descriptions": format_term_descriptions(
                state.get("domain_term_mappings", {})
            ),
            "failed_sql": generated_sql.get("permission_controlled_sql", ""),
            "error_message": execution_result["error"],
        }

        # 创建并执行错误分析链
        analysis_chain = create_error_analysis_chain()
        result = analysis_chain.invoke(input_data)
        
        logger.info(f"错误分析结果: 是否可修复={result['is_sql_fixable']}")
        if result['is_sql_fixable'] and result['fixed_sql']:
            logger.info(f"修复后的SQL: {result['fixed_sql']}")

        # 构造返回结果
        response = {
            "error_analysis_result": {
                "is_sql_fixable": result["is_sql_fixable"],
                "error_analysis": result["error_analysis"],
                "fixed_sql": result["fixed_sql"] if result["is_sql_fixable"] else None,
                "user_feedback": result["user_feedback"]
            }
        }

        # 只有在有用户反馈时才添加消息
        if result.get("user_feedback"):
            response["messages"] = [AIMessage(content=result["user_feedback"])]

        return response

    except Exception as e:
        error_msg = f"错误分析过程出错: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
