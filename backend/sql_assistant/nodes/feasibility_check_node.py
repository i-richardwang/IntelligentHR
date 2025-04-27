"""
查询可行性检查节点模块。
负责评估现有数据表是否能满足用户的查询需求。
"""

from pydantic import BaseModel, Field
from typing import Optional
import logging

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from backend.sql_assistant.utils.format_utils import (
    format_table_structures,
    format_term_descriptions,
)
from utils.llm_tools import init_language_model, LanguageModelChain
from langchain.schema import AIMessage

logger = logging.getLogger(__name__)


class FeasibilityCheckResult(BaseModel):
    """数据源匹配度评估结果

    评估给定的数据表结构是否包含回答用户查询所需的必要信息
    """

    feasible_analysis: str = Field(
        ..., description="评估判断的详细思考过程"
    )
    is_feasible: bool = Field(
        ..., description="表示现有数据表是否包含回答用户查询所需的全部必要信息"
    )
    user_feedback: Optional[str] = Field(
        None, description="当数据表无法满足查询需求时，面向用户的友好反馈信息"
    )


FEASIBILITY_CHECK_SYSTEM_PROMPT = """你是一位专业的数据分析专家,需要严格评估数据表是否能提供完整且准确的信息来回答用户查询。

评估步骤:

1. 分析数据表的基本属性
   - 理解数据表的主要用途(如:交易记录、培训记录、员工花名册等)
   - 判断数据表的更新特征(如:快照、流水、主数据等)
   - 评估数据表的覆盖范围(如:是否包含所有员工、是否仅包含特定群体)

2. 分析用户查询的数据需求
   - 识别查询需要的核心信息(如:人数、金额、状态等)
   - 确定查询的业务范围(如:全量员工、特定部门、特定时间段)
   - 判断查询是否需要准确的时点数据

3. 严格的匹配度评估
   必须满足所有以下条件才能判定为匹配(is_feasible = true):
   - 数据表的主要用途与查询目标直接相关
   - 数据表包含查询所需的完整信息
   - 数据表的覆盖范围满足查询需求
   - 数据表能提供准确的结果(不是部分数据或推导数据)
   
4. 常见的不匹配情况
   以下情况必须判定为不匹配(is_feasible = false):
   - 数据表的统计口径与查询需求不符
   - 数据表虽有相关字段但无法保证数据完整性
   - 需要通过复杂推导才能得到查询结果

5. 输出要求：
   a) feasible_analysis需要包含:
      - 识别用户查询的核心数据需求
      - 分析现有数据表的关键信息
      - 说明匹配或不匹配的主要原因
      - 保持简洁的思考过程记录
   
   b) 当判定为不匹配时，user_feedback需要:
      - 简明扼要地告知用户原因
      - 区分两种情况：
        * 数据库中没有用户查询相关的数据表
        * 存在相关表但缺少必要的字段信息
      - 使用自然语言描述表名和字段，不暴露数据库表名
"""


FEASIBILITY_CHECK_USER_PROMPT = """请评估以下数据表是否包含足够信息来回答用户查询:

1. 用户查询需求：
{rewritten_query}

2. 现有数据表结构:
{table_structures}

3. 相关业务术语解释(如果存在)：
{term_descriptions}

请首先理解数据表的基本属性和覆盖范围,然后严格评估是否能提供准确的查询结果。
即使数据表包含相关字段,也要判断数据的完整性和准确性是否满足查询需求。"""


def create_feasibility_check_chain(temperature: float = 0.0) -> LanguageModelChain:
    """创建可行性检查任务链"""
    llm = init_language_model(temperature=temperature)

    return LanguageModelChain(
        model_cls=FeasibilityCheckResult,
        sys_msg=FEASIBILITY_CHECK_SYSTEM_PROMPT,
        user_msg=FEASIBILITY_CHECK_USER_PROMPT,
        model=llm,
    )()


def feasibility_check_node(state: SQLAssistantState) -> dict:
    """查询可行性检查节点函数"""
    if not state.get("rewritten_query"):
        return {"error": "状态中未找到改写后的查询"}
    if not state.get("table_structures"):
        return {"error": "状态中未找到表结构信息"}

    try:
        input_data = {
            "rewritten_query": state["rewritten_query"],
            "table_structures": format_table_structures(state["table_structures"]),
            "term_descriptions": format_term_descriptions(
                state.get("domain_term_mappings", {})
            ),
        }

        check_chain = create_feasibility_check_chain()
        result = check_chain.invoke(input_data)
        
        logger.info(f"可行性检查结果: {'通过' if result['is_feasible'] else '不通过'}" + 
                   (f", 原因: {result['user_feedback']}" if not result['is_feasible'] else ""))

        response = {
            "feasibility_check": {
                "is_feasible": result["is_feasible"],
                "feasible_analysis": result["feasible_analysis"],
                "user_feedback": result["user_feedback"] if not result["is_feasible"] else None,
            }
        }

        if not result["is_feasible"] and result["user_feedback"]:
            response["messages"] = [AIMessage(content=result["user_feedback"])]

        return response

    except Exception as e:
        error_msg = f"可行性检查过程出错: {str(e)}"
        return {"error": error_msg}
