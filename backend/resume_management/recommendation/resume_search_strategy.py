import asyncio
import uuid
import os
from typing import List, Dict, Optional
from pydantic import BaseModel
from langfuse.callback import CallbackHandler
from backend_demo.resume_management.recommendation.recommendation_state import (
    ResumeSearchStrategy,
    CollectionSearchStrategy,
)
from utils.llm_tools import LanguageModelChain, init_language_model

# 初始化语言模型
language_model = init_language_model(
    provider=os.getenv("SMART_LLM_PROVIDER"), model_name=os.getenv("SMART_LLM_MODEL")
)


class ResumeSearchStrategyGenerator:
    """生成整体简历搜索策略的类"""

    def __init__(self):
        self.system_message = """
        你是一个智能简历推荐系统的搜索策略生成器。你的任务是分析用户的查询需求，确定哪些 Milvus 数据库中的简历数据 collection 与查询最相关，并为每个相关的 collection 分配一个相关性分数。

        Milvus 数据库中可用的简历数据 collection 及其描述如下：
        1. work_experiences: 工作经历集合，包含岗位名称和职责描述的向量表示，用于匹配相似的工作经验。
        2. educations: 教育背景集合，包含学位和专业的向量表示，用于匹配相关的教育经历。
        3. personal_infos: 个人总结集合，包含个人概述的向量表示，用于综合匹配候选人背景。
        4. project_experiences: 项目经历集合，包含项目名称和详情的向量表示，用于匹配相关的项目经验。
        5. skills: 技能集合，包含技能的向量表示，用于精确匹配特定的技能需求。

        请根据用户查询内容，生成一个 Milvus collection 相关性列表，每个项目包含：
        - collection_name: Milvus collection 的名称
        - relevance_score: 相关性分数（0到1之间的浮点数）

        在分析用户需求和分配相关性分数时，请遵循以下原则：

        1. 优先考虑工作经历（work_experiences）：通常 work_experiences collection 应该获得较高的相关性分数，因为它最直接反映了候选人的实际工作能力和经验。
        2. 识别核心需求：仔细分析用户查询，辨别出真正的核心需求。聚焦在职位的本质要求上，而不是过分关注通用或辅助性的技能。
        3. 平衡考虑各个方面：根据用户的具体需求对所有相关 collection 进行全面评估。某些特定职位可能需要更看重其他 collection，如教育背景或项目经验。
        4. 考虑隐含需求：除了明确表述的要求，也要考虑职位可能隐含的需求。
        5. 权重：评估各维度对于整体职位需求的重要性，而不仅仅是因为提到了某个维度就给予过高的权重。
        6. 向量相似度考虑：记住 Milvus 是一个向量数据库，我们的查询策略应该考虑如何有效利用向量相似度搜索。
        7. 查询效率：考虑如何平衡查询的精确度和速度。有时，重点关注少数最相关的 collection 可能比尝试匹配所有 collection 更有效。
        8. 整体匹配度：考虑如何通过不同 collection 的组合来最好地反映出理想候选人的整体形象。

        请确保所有相关性分数之和为1。根据查询的重点和每个 collection 的相关性，合理分配分数。仔细考虑查询中明确提到的要求和可能隐含的需求，以决定哪些 collection 更重要。你的目标是创建一个能够最准确地匹配用户真实需求的 Milvus 搜索策略。
        """

        self.human_message_template = """
        用户查询：
        {refined_query}

        请生成简历数据集合相关性列表，确保相关性分数之和为1，并以JSON格式输出。请全面分析查询，考虑所有明确和隐含的需求。
        """

        self.resume_search_strategy_chain = LanguageModelChain(
            ResumeSearchStrategy,
            self.system_message,
            self.human_message_template,
            language_model,
        )()

    def create_langfuse_handler(self, session_id: str, step: str) -> CallbackHandler:
        """创建 Langfuse 回调处理器"""
        return CallbackHandler(
            tags=["resume_search_strategy"],
            session_id=session_id,
            metadata={"step": step},
        )

    async def generate_resume_search_strategy(
        self, refined_query: str, session_id: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """
        生成整体的简历搜索策略。

        Args:
            refined_query (str): 精炼后的用户查询
            session_id (Optional[str]): 会话ID

        Returns:
            List[Dict[str, float]]: 包含集合名称和相关性分数的字典列表

        Raises:
            ValueError: 如果精炼后的查询为空
        """
        if not refined_query:
            raise ValueError("精炼后的查询不能为空。无法生成搜索策略。")

        session_id = session_id or str(uuid.uuid4())
        langfuse_handler = self.create_langfuse_handler(
            session_id, "generate_search_strategy"
        )

        search_strategy_result = await self.resume_search_strategy_chain.ainvoke(
            {"refined_query": refined_query}, config={"callbacks": [langfuse_handler]}
        )

        resume_search_strategy = ResumeSearchStrategy(**search_strategy_result)

        collection_relevances = [
            {
                "collection_name": cr.collection_name,
                "relevance_score": cr.relevance_score,
            }
            for cr in resume_search_strategy.collection_relevances
        ]

        print("已生成初步的简历匹配策略")

        return collection_relevances


class CollectionSearchStrategyGenerator:
    """生成特定集合搜索策略的类"""

    def __init__(self):
        self.collection_specific_instructions = {
            "work_experiences": {
                "field_descriptions": """
                - position_vector: 职位名称的向量表示
                - responsibilities_vector: 工作职责的向量表示
                """,
                "content_generation_guidelines": """
                对于position_vector:
                - query_content应该是从用户需求中提取的精炼的岗位名称。
                - 避免在一个查询策略中包含多个不同方面的岗位名称。如果有多个相关的岗位名称可供检索，应该生成多个查询策略。
                - 例如：如果用户寻找"数据分析师或数据科学家"，应该分别为"数据分析师"和"数据科学家"生成两个查询策略。

                对于responsibilities_vector:
                - query_content应该是一段不少于100字的职责描述。
                - 如果用户没有提供详细的岗位说明，应该基于用户的意图和行业常见职责，模拟一段合理的工作职责描述。
                - 如果用户提供的工作职责较短，应该根据行业标准和常见实践加以丰富和完善。
                - 确保描述涵盖用户提到的所有关键技能和经验要求。
                """,
            },
            "educations": {
                "field_descriptions": """
                - degree_vector: 学位的向量表示
                - major_vector: 专业的向量表示
                """,
                "content_generation_guidelines": """
                对于degree_vector:
                - query_content应该是从用户需求中提取的精确学位名称。
                - 如果用户没有明确指定学位，可以根据职位要求推断合适的学位级别（如学士、硕士、博士）。
                - 可以包括多个相关的学位，每个学位对应一个单独的查询策略。
                
                对于major_vector:
                - query_content应该是具体的专业名称或相关学科领域。
                - 如果用户提供的专业信息较为宽泛，应该列出可能的具体专业方向。
                - 考虑包括主修和辅修专业，如果相关的话。
                - 可以生成多个查询策略来覆盖相关的专业领域。
                """,
            },
            "skills": {
                "field_descriptions": """
                - skill_vector: 技能的向量表示
                """,
                "content_generation_guidelines": """
                对于skill_vector:
                - 除非用户查询中明确枚举或要求超过3项以上的技能，否则最多生成三个用于检索的技能。
                - 根据技能对满足用户需求的重要性进行选择和排序。
                - query_content应该是具体、明确的技能名称或术语。
                - 避免使用过于宽泛或模糊的技能描述。
                """,
            },
            "project_experiences": {
                "field_descriptions": """
                - name_vector: 项目名称的向量表示
                - details_vector: 项目详情的向量表示
                """,
                "content_generation_guidelines": """
                对于name_vector:
                - query_content应该是简洁但信息丰富的项目名称或类型。
                - 项目名称应反映用户查询中提到的关键技能、技术或领域。
                - 如果用户没有明确指定项目类型，可以根据职位要求推断可能相关的项目。
                
                对于details_vector:
                - query_content应该是一段全面的项目描述，通常不少于100字。
                - 描述应涵盖项目的关键方面，包括：
                  a. 项目目标和背景
                  b. 使用的技术和工具
                  c. 个人职责和贡献
                  d. 项目成果和影响
                  e. 遇到的挑战和解决方案
                  f. 与用户查询相关的特定技能或经验展示
                """,
            },
            "personal_infos": {
                "field_descriptions": """
                - summary_vector: 个人总结/概述的向量表示
                """,
                "content_generation_guidelines": """
                对于summary_vector:
                - query_content应该是一段全面的个人概述，通常不少于100字。
                - 概述应该涵盖用户查询中提到的关键要素，如职位、技能、经验水平、行业背景等。
                - 如果用户没有提供详细的要求，应该基于用户的意图和行业标准，创建一个理想候选人的概述。
                - 确保概述涵盖以下方面（如相关）：
                  a. 专业身份或目标职位
                  b. 核心技能和专长
                  c. 行业经验和成就
                  d. 教育背景
                  e. 个人特质或软技能
                """,
            },
        }

        self.collection_specific_system_message_template = """
        你是一个智能简历推荐系统的搜索策略生成器。你的任务是分析用户的查询需求，并为{collection_name}集合中的向量字段生成精确的查询策略。

        {collection_name}集合的向量字段如下：

        {field_descriptions}

        请严格遵循以下指南：

        1. 仔细分析用户查询，理解其明确和潜在的需求，包括所需的职位、技能、经验等。

        2. 为每个相关的向量字段生成一个或多个查询策略。每个查询策略都应包含field_name、query_content和relevance_score。

        3. 生成query_content时，请遵循以下原则：
           {content_generation_guidelines}

        4. 在设计query_content时，请站在招聘者的角度思考，设计能够匹配出最符合用户需求的简历的查询文本。考虑以下因素：
           - 所需的核心技能和次要技能
           - 经验水平（如初级、中级、高级）
           - 特定的行业知识或背景
           - 可能的工作职责和日常任务
           - 项目经验或专业领域

        5. 为每个查询分配一个relevance_score，表示该查询对于满足用户需求的重要性。确保：
           - 所有查询的relevance_score之和为1。
           - 根据查询的重点和每个方面的重要性，合理分配分数。

        6. 如果用户的查询非常具体，可以生成多个针对不同向量字段的查询策略，每个策略侧重于不同的方面，以全面覆盖用户的需求。

        7. 确保生成的查询策略全面且准确地反映了用户的需求，包括明确表达的要求和可能的隐含需求。
        """

        self.collection_specific_human_message_template = """
        用户查询：
        {refined_query}

        请针对{collection_name}集合中的向量字段生成查询策略列表。确保所有查询的相关性分数之和为1，并严格按照指定的JSON格式输出。请记住，你的目标是生成能够匹配出最符合用户需求的简历的查询策略。

        **query_content必须使用英文！**
        """

    def create_langfuse_handler(self, session_id: str, step: str) -> CallbackHandler:
        """创建 Langfuse 回调处理器"""
        return CallbackHandler(
            tags=["collection_search_strategy"],
            session_id=session_id,
            metadata={"step": step},
        )

    async def generate_single_collection_strategy(
        self,
        collection_name: str,
        refined_query: str,
        session_id: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, CollectionSearchStrategy]:
        """为单个集合生成搜索策略"""
        async with semaphore:
            system_message = self.collection_specific_system_message_template.format(
                collection_name=collection_name,
                field_descriptions=self.collection_specific_instructions[
                    collection_name
                ]["field_descriptions"],
                content_generation_guidelines=self.collection_specific_instructions[
                    collection_name
                ]["content_generation_guidelines"],
            )

            collection_search_strategy_chain = LanguageModelChain(
                CollectionSearchStrategy,
                system_message,
                self.collection_specific_human_message_template,
                language_model,
            )()

            langfuse_handler = self.create_langfuse_handler(
                session_id, f"generate_strategy_for_{collection_name}"
            )
            search_strategy_result = await collection_search_strategy_chain.ainvoke(
                {"refined_query": refined_query, "collection_name": collection_name},
                config={"callbacks": [langfuse_handler]},
            )

            return {collection_name: CollectionSearchStrategy(**search_strategy_result)}

    async def generate_collection_search_strategy(
        self,
        refined_query: str,
        collection_relevances: List[Dict[str, float]],
        session_id: Optional[str] = None,
    ) -> Dict[str, CollectionSearchStrategy]:
        """
        并行为每个相关的集合生成详细的搜索策略。

        Args:
            refined_query (str): 精炼后的用户查询
            collection_relevances (List[Dict[str, float]]): 集合相关性列表
            session_id (Optional[str]): 会话ID

        Returns:
            Dict[str, CollectionSearchStrategy]: 每个集合的搜索策略

        Raises:
            ValueError: 如果精炼后的查询为空或集合相关性列表为空
        """
        if not refined_query or not collection_relevances:
            raise ValueError(
                "精炼后的查询或集合相关性列表不能为空。无法生成集合搜索策略。"
            )

        session_id = session_id or str(uuid.uuid4())
        semaphore = asyncio.Semaphore(3)  # 限制并发数为3

        tasks = [
            self.generate_single_collection_strategy(
                collection_info["collection_name"], refined_query, session_id, semaphore
            )
            for collection_info in collection_relevances
        ]

        results = await asyncio.gather(*tasks)
        collection_search_strategies = {}
        for result in results:
            collection_search_strategies.update(result)

        print("已完成详细的简历搜索策略制定")
        return collection_search_strategies
