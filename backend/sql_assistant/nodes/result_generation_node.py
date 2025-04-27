"""
结果生成节点模块。
负责将SQL查询结果转化为用户友好的描述。
"""

import logging
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from backend_demo.sql_assistant.states.assistant_state import SQLAssistantState
from backend_demo.sql_assistant.utils.format_utils import (
    format_results_preview,
    format_term_descriptions,
    format_full_results
)
from utils.llm_tools import init_language_model, LanguageModelChain

logger = logging.getLogger(__name__)


class ResultGenerationOutput(BaseModel):
    """查询结果生成输出模型"""
    result_description: str = Field(
        ...,
        description="面向用户的查询结果描述"
    )


# 系统提示词，保持原有定义
RESULT_GENERATION_SYSTEM_PROMPT = """你是一个专业的数据分析师，负责将SQL查询结果转化为用户友好的自然语言描述。
请遵循以下规则生成反馈：

1. 反馈内容要求：
   - 直接回答用户的查询需求
   - 使用简洁、专业的语言
   - 适当解释异常或特殊数据
   
2. 特殊情况处理策略：

   A. 大数据集（row_count > 20）
      处理要求：
      - 告知用户数据量较大，无法得知和展示明细数据
      - 建议用户：
        * 添加更多过滤条件缩小范围
        * 按维度分组后查询统计值
      - 提供优化示例

   B. 空结果集（row_count = 0）
      分析用户查询意图：

      1) 期望型查询（用户预期存在数据）
         适用场景：
         - 询问数量、统计值（如"有多少"、"总共"、"平均"等）
         - 查询分布情况（如"销售情况"、"参与度"等）
         - 查询具体数值（如"工资是多少"、"评分是多少"等）
         
         处理要求：
         - 清晰说明使用的查询条件：
           * 数据源/表名（用自然语言表示表名，不要泄露数据库的表名）
           * 过滤条件
           * 匹配关键词
         - 提示潜在原因：
           * 字段名/值是否有其他标准称谓
           * 是否需要模糊匹配
           * 时间范围是否准确
         - 建议上层查询：
           * 提供更宽泛的查询建议
           * 引导查看相关分布情况
      
      2) 验证型查询（结果可能为空）
         适用场景：
         - 是否类问题（如"是否参与"、"是否担任"等）
         - 存在性判断（如"有没有"、"是否存在"等）
         - 真伪判断（如"是不是"、"是否为"等）
         
         处理要求：
         - 明确说明查询条件
         - 直接给出否定答复
         - 无需额外引导
      
3. 注意事项：
   - 使用正确的业务术语
   - 避免过度解读数据
   - 保持客观中立的语气
"""

RESULT_GENERATION_USER_PROMPT = """请根据以下信息生成查询结果描述：

1. 用户的查询：
{rewritten_query}

2. 查询结果：
总行数：{row_count}
是否截断：{truncated}
查询数据源：{data_source}
查询语句：```{sql_query}```
数据预览：
{results_preview}

3. 业务术语说明：
{term_descriptions}

请生成一段清晰、专业的描述，向用户说明查询结果。"""


def create_result_generation_chain(temperature: float = 0.0) -> LanguageModelChain:
    """创建结果生成任务链

    Args:
        temperature: 模型温度参数，控制输出的随机性

    Returns:
        LanguageModelChain: 配置好的结果生成任务链
    """
    llm = init_language_model(temperature=temperature)

    return LanguageModelChain(
        model_cls=ResultGenerationOutput,
        sys_msg=RESULT_GENERATION_SYSTEM_PROMPT,
        user_msg=RESULT_GENERATION_USER_PROMPT,
        model=llm,
    )()


def result_generation_node(state: SQLAssistantState) -> dict:
    """结果生成节点函数

    将SQL查询结果转化为用户友好的自然语言描述。
    包括数据概述、关键指标和特殊情况说明。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含结果描述的状态更新
    """
    # 获取必要信息
    execution_result = state.get("execution_result", {})
    if not execution_result or not execution_result.get('success'):
        return {"error": "状态中未找到成功的执行结果"}

    try:
        # 准备输入数据
        generated_sql = state.get("generated_sql", {})
        input_data = {
            "rewritten_query": state["rewritten_query"],
            "row_count": execution_result["row_count"],
            "truncated": execution_result["truncated"],
            "results_preview": format_results_preview(execution_result),
            "term_descriptions": format_term_descriptions(
                state.get("domain_term_mappings", {})
            ),
            "data_source": state.get("matched_tables", [{}])[0].get("table_name", "未知数据源"),
            "sql_query": generated_sql.get("sql_query", "未知SQL查询")
        }

        # 创建并执行结果生成链
        generation_chain = create_result_generation_chain()
        result = generation_chain.invoke(input_data)
        
        logger.info(f"结果生成完成: {result['result_description'][:100]}...")

        # 将结果描述和完整表格结果组合在一起
        formatted_table = format_full_results(execution_result)
        combined_message = (
            f"{result['result_description']}\n\n"
            f"查询结果详情：\n"
            f"{formatted_table}"
        )

        return {
            "result_description": result["result_description"],
            "messages": [AIMessage(content=result['result_description'])]
        }

    except Exception as e:
        error_msg = f"结果生成过程出错: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
