"""
查询改写节点模块。
负责将用户的原始查询改写为规范化的形式。
"""

from pydantic import BaseModel, Field
import logging

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from backend.sql_assistant.utils.format_utils import (
    format_conversation_history,
    format_term_descriptions
)
from utils.llm_tools import init_language_model, LanguageModelChain

logger = logging.getLogger(__name__)


class QueryRewrite(BaseModel):
    """查询需求改写结果模型"""
    rewritten_query: str = Field(
        ...,
        description="改写后的查询语句"
    )


# 系统提示词，保持原有定义
QUERY_REWRITE_SYSTEM_PROMPT = """你是一个专业的数据分析师，负责将用户的数据查询需求改写为更规范、明确的形式。
请遵循以下规则改写查询：

改写原则：
1. 保持查询的核心意图不变，关注用户表达的查询需求，不要将助手的引导性问题加入改写结果。
2. 使用检索到的标准的业务术语替换原始不标准的术语（如果存在）
3. 明确指出查询的数据范围（时间、地区等）
4. 明确标注所有查询条件的归属关系
5. 规范化条件表述
6. 移除无关的修饰词和语气词

输出要求：
- 输出一个完整的查询句子
- 使用陈述句形式
- 保持语言简洁明确
- 确保包含所有必要的查询条件"""

QUERY_REWRITE_USER_PROMPT = """请根据以下信息改写用户的查询需求：

1. 对话历史：
{dialogue_history}

2. 检索到的业务术语信息(如果存在)：
{term_descriptions}

请按照系统消息中的规则改写查询，并以指定的JSON格式输出结果。"""


def create_query_rewrite_chain(temperature: float = 0.0) -> LanguageModelChain:
    """创建查询改写任务链

    Args:
        temperature: 模型温度参数，控制输出的随机性

    Returns:
        LanguageModelChain: 配置好的查询改写任务链
    """
    llm = init_language_model(temperature=temperature)

    return LanguageModelChain(
        model_cls=QueryRewrite,
        sys_msg=QUERY_REWRITE_SYSTEM_PROMPT,
        user_msg=QUERY_REWRITE_USER_PROMPT,
        model=llm,
    )()


def query_rewrite_node(state: SQLAssistantState) -> dict:
    """查询需求改写节点函数

    将用户的原始查询改写为规范化的形式，
    使用标准的业务术语，明确查询条件和范围。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含改写后查询的状态更新
    """
    # 获取对话历史
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("状态中未找到消息历史")

    # 格式化对话历史
    dialogue_history = format_conversation_history(messages)

    # 格式化术语解释
    term_descriptions = format_term_descriptions(
        state.get("domain_term_mappings", {})
    )

    # 创建改写链
    rewrite_chain = create_query_rewrite_chain()

    # 执行改写
    result = rewrite_chain.invoke({
        "dialogue_history": dialogue_history,
        "term_descriptions": term_descriptions
    })
    
    logger.info(f"查询改写完成，改写结果: {result['rewritten_query']}")

    # 更新状态
    return {
        "rewritten_query": result["rewritten_query"]
    }
