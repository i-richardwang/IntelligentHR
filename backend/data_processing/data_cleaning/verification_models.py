import os
from typing import Literal
from pydantic import BaseModel, Field

from utils.llm_tools import init_language_model, LanguageModelChain

# 初始化语言模型
language_model = init_language_model(
    provider=os.getenv("SMART_LLM_PROVIDER"), model_name=os.getenv("SMART_LLM_MODEL")
)


class InputValidation(BaseModel):
    """用户输入的验证结果。"""

    is_valid: bool = Field(
        ..., description="指示用户输入是否为有效的标签（'True'）或不是（'False'）。"
    )


class EntityRecognition(BaseModel):
    """从网络搜索结果中识别实体名称的结果。"""

    identified_entity: str = Field(
        ..., description="从搜索结果中提取的标准化实体名称。"
    )
    recognition_status: Literal["known", "unknown"] = Field(
        ..., description="指示实体名称是否成功识别（'known'）或未识别（'unknown'）。"
    )


class EntityVerification(BaseModel):
    """检索到的实体名称与用户查询匹配的验证结果。"""

    verification_status: Literal["verified", "unverified"] = Field(
        ...,
        description="指示检索到的实体名称是否与用户的查询相匹配（'verified'）或不匹配（'unverified'）。",
    )


# 定义系统消息
system_message_validation = """
你的任务是判断用户输入的查询{entity_type}是否是一个具体的、有效的{entity_type}。
你将收到一个用户查询的{entity_type}。
请判断该名称是否足以识别一个特定的{entity_type}。
如果名称具体且明确，足以识别一个特定的{entity_type}，则将其标记为'True'。
如果名称模糊，或者不是指一个特定的{entity_type}，则标记为'False'。
{validation_instructions}
"""

system_message_search_analysis = """
你的工作是根据提供的网络搜索结果来判断搜索结果是否与查询的{entity_type}相关。
如果搜索结果与查询的{entity_type}相关，你需要进一步分析并提取标准化的{entity_type}。
{analysis_instructions}
如果能够提取出标准化的{entity_type}，请提供该标准化名称，并标记状态为'known'。
如果无法提取出相关信息，或者搜索结果与查询的{entity_type}不相关，请标记状态为'unknown'，并省略{entity_type}名称。
在进行判断时，请考虑{entity_type}名称的相似性、搜索结果中提及的相关信息等。
"""

system_message_verification = """
你的任务是验证从向量数据库检索到的{entity_type}是否与用户提供的查询{entity_type}指向同一个标准化{entity_type}。
你将收到一个用户查询的{entity_type}、从数据库检索到的最相似的{entity_type}以及相关的搜索结果。
请结合搜索结果来判断检索到的{entity_type}是否确实是指向用户查询的同一个标准化{entity_type}。
{verification_instructions}
如果检索到的{entity_type}与搜索结果一致，并且可以确认是指向同一个标准化{entity_type}，则将其标记为'verified'。
如果无法确认是同一标准化{entity_type}，或者搜索结果与查询不相关，则标记为'unverified'。
在判断时，请考虑{entity_type}名称的正式表述、可能的简称或全称差异，以及搜索结果中的相关信息。
"""

# 定义人类消息
human_message_validation = "用户查询的{entity_type}：{user_query}"

human_message_search_analysis = """
针对特定查询'{user_query}'，请仔细评估以下网络搜索片段：

{snippets}

如果搜索结果明确指向查询中的{entity_type}，并且您能够提取出标准化的{entity_type}名称，
请提供该标准化名称并标注状态为'known'。
如果搜索结果不明确或无法提取出相关信息，请标注状态为'unknown'，并省略{entity_type}名称。"""

human_message_verification = """
用户查询的{entity_type}：{user_query} \n
检索到的{entity_type}：{retrieved_name} \n
搜索结果：
{search_results}"""

# 创建处理链
input_validator = LanguageModelChain(
    InputValidation, system_message_validation, human_message_validation, language_model
)()

search_analysis = LanguageModelChain(
    EntityRecognition,
    system_message_search_analysis,
    human_message_search_analysis,
    language_model,
)()

name_verifier = LanguageModelChain(
    EntityVerification,
    system_message_verification,
    human_message_verification,
    language_model,
)()
