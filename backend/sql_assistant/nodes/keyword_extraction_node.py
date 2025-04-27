"""
关键词提取节点模块。
负责从用户查询中提取关键业务实体和术语。
"""

from pydantic import BaseModel, Field
from typing import List
import logging

from backend.sql_assistant.states.assistant_state import SQLAssistantState
from backend.sql_assistant.utils.format_utils import format_conversation_history
from utils.llm_tools import init_language_model, LanguageModelChain

logger = logging.getLogger(__name__)


class QueryKeywordExtraction(BaseModel):
    """关键词提取结果模型"""
    keywords: List[str] = Field(
        default_factory=list,
        description="从查询中提取的关键实体列表"
    )


# 系统提示词，保持原有定义
DOMAIN_ENTITY_EXTRACTION_PROMPT = """你是一个专业的数据分析师，负责从用户查询中提取需要进行准确匹配的具体业务实体名称。

提取原则：
1. 只提取那些在数据库中需要进行精确匹配的具体实体名称
2. 这些实体通常具有唯一性和特定性，替换成其他名称会导致完全不同的查询结果

不要提取的内容：
1. 泛化的概念词（如：培训、数据、人员、内容等）
2. 时间相关表述（如：2023年、上个月等）
3. 模糊的描述词（如：最近的、所有的等）
4. 常见度量单位（如：数量、金额、比例等）

判断标准：
- 这个词是否需要与数据库中的具体值进行精确匹配
- 如果用其他词替代，查询结果是否会完全不同
- 这个词是否代表了一个独特的业务实体

输出要求：
1. 只输出真正需要进行精确匹配的实体名称
2. 如果没有找到需要精确匹配的实体，返回空列表
3. 去除重复项
4. 保持实体的原始表述形式"""

KEYWORD_EXTRACTION_USER_PROMPT = """请从以下对话中提取需要进行精确匹配的具体业务实体名称：

对话记录：
{dialogue_history}

请按照系统消息中的规则提取关键实体，并以指定的JSON格式输出结果。
注意：只提取那些需要与数据库中的具体值进行精确匹配的实体名称。"""


def create_keyword_extraction_chain(temperature: float = 0.0) -> LanguageModelChain:
    """创建关键词提取任务链

    Args:
        temperature: 模型温度参数，控制输出的随机性

    Returns:
        LanguageModelChain: 配置好的关键词提取任务链
    """
    llm = init_language_model(temperature=temperature)
    return LanguageModelChain(
        model_cls=QueryKeywordExtraction,
        sys_msg=DOMAIN_ENTITY_EXTRACTION_PROMPT,
        user_msg=KEYWORD_EXTRACTION_USER_PROMPT,
        model=llm,
    )()


def keyword_extraction_node(state: SQLAssistantState) -> dict:
    """关键词提取节点函数

    从用户查询和对话历史中提取关键的业务实体和术语。
    包括业务对象、指标和维度等关键信息。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含提取的关键词的状态更新
    """
    # 获取对话历史
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("状态中未找到消息历史")

    # 格式化对话历史
    dialogue_history = format_conversation_history(messages)

    # 创建提取链
    extraction_chain = create_keyword_extraction_chain()

    # 执行提取
    result = extraction_chain.invoke({
        "dialogue_history": dialogue_history
    })
    
    logger.info(f"提取到的关键词: {result['keywords']}")

    # 更新状态
    return {
        "keywords": result["keywords"]
    }
