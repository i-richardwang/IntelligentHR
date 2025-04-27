"""
意图分析节点模块。
负责分析用户查询意图的清晰度，判断是否需要进一步澄清。
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
import logging

from backend_demo.sql_assistant.states.assistant_state import SQLAssistantState
from backend_demo.sql_assistant.utils.format_utils import format_conversation_history
from utils.llm_tools import init_language_model, LanguageModelChain

logger = logging.getLogger(__name__)


class QueryIntentAnalysis(BaseModel):
    """查询意图分析结果模型"""
    is_intent_clear: bool = Field(
        ...,
        description="查询意图是否明确。True表示意图清晰，可以继续处理；False表示需要澄清"
    )
    clarification_question: Optional[str] = Field(
        None,
        description="当意图不明确时，需要向用户提出的澄清问题"
    )


INTENT_CLARITY_ANALYSIS_PROMPT = """
你是一个专业的数据查询助手，负责判断用户数据查询请求的可执行性。请以开放和务实的态度分析用户的查询意图。

判断标准:
1. 查询目标明确：清楚用户想要查询什么数据
2. 查询条件完整：如果需要筛选，条件是否明确(非必要)
3. 时间范围明确：如果查询涉及时间，时间范围是否明确(非必要)
4. 允许未知的专有名词和领域术语的存在，这些将在后续步骤中得到解释
5. 对模糊但可推断的查询持包容态度

输出要求:
1. 如果不是数据查询请求，设置is_intent_clear为false，并生成回复说明你只能处理数据查询相关的问题
2. 如果意图明确，设置is_intent_clear为true
3. 如果意图不明确，设置is_intent_clear为false，并生成一个明确的问题来获取缺失信息
4. 问题要具体指出缺失的信息点，便于用户理解和回答
"""

INTENT_ANALYSIS_USER_PROMPT = """请分析以下用户的数据查询请求：

用户查询：
{query}

请根据系统指令中的要求，按照指定的JSON格式输出分析结果。"""


def create_intent_clarity_analyzer(temperature: float = 0.0) -> LanguageModelChain:
    """创建意图清晰度分析器

    构建用于评估查询意图清晰度的LLM链

    Args:
        temperature: 模型温度参数，控制输出的随机性

    Returns:
        LanguageModelChain: 配置好的意图清晰度分析链
    """
    llm = init_language_model(temperature=temperature)
    return LanguageModelChain(
        model_cls=QueryIntentAnalysis,
        sys_msg=INTENT_CLARITY_ANALYSIS_PROMPT,
        user_msg=INTENT_ANALYSIS_USER_PROMPT,
        model=llm,
    )()


def intent_analysis_node(state: SQLAssistantState) -> dict:
    """分析用户查询意图的节点函数

    分析用户的查询请求，判断是否包含足够的信息来执行查询。
    如果意图不明确，会生成澄清问题。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含意图分析结果的状态更新
    """
    # 获取所有对话历史
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("状态中未找到消息历史")

    # 格式化对话历史
    dialogue_history = format_conversation_history(messages)

    # 创建分析链
    analysis_chain = create_intent_clarity_analyzer()

    # 执行分析
    result = analysis_chain.invoke({"query": dialogue_history})
    logger.info(f"意图分析结果: 意图清晰度={result['is_intent_clear']}, 澄清问题={result.get('clarification_question')}")

    # 如果意图不明确，添加一个助手消息询问澄清
    response = {}
    if not result["is_intent_clear"] and result.get("clarification_question"):
        response["messages"] = [
            AIMessage(content=result["clarification_question"])]

    # 更新状态
    response.update({
        "query_intent": {
            "is_clear": result["is_intent_clear"],
            "clarification_needed": not result["is_intent_clear"],
            "clarification_question": result["clarification_question"] if not result["is_intent_clear"] else None
        },
        "is_intent_clear": result["is_intent_clear"]
    })

    return response
