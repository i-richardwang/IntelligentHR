import os
import logging
import pandas as pd
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import asyncio

from utils.llm_tools import create_embeddings
from utils.vector_db_utils import (
    connect_to_milvus,
    get_num_entities,
    search_by_vector_column,
)

logger = logging.getLogger(__name__)

# 检索 limit 的合理上限：空集合时 limit 必须 >=1；上限只是防御性封顶（libSQL 暴力检索
# 本身无硬限制，demo 规模也远达不到）。
SEARCH_TOPK_LIMIT = 16384


def clamp_search_limit(num_entities: int) -> int:
    """将检索 limit 收敛到合法区间 [1, SEARCH_TOPK_LIMIT]。"""
    return min(max(num_entities, 1), SEARCH_TOPK_LIMIT)


class ResumeScorer:
    """简历评分器，负责计算简历的得分"""

    def __init__(self):
        # 统一走项目 embedding 工厂：端点 / 密钥 / 模型均来自环境变量，
        # 不再硬编码供应商 base_url，保证与全项目 embedding 口径一致
        self.embeddings = create_embeddings()
        self.db_name = os.getenv("VECTOR_DB_DATABASE_RESUME", "resume")

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        异步获取文本的嵌入向量。

        Args:
            text (str): 输入文本

        Returns:
            Optional[List[float]]: 嵌入向量，如果出现错误则返回 None
        """
        max_retries = 3
        for _ in range(max_retries):
            try:
                embedding = await self.embeddings.aembed_query(text)
                await asyncio.sleep(0.07)  # 避免频繁请求
                return embedding
            except Exception as e:
                logger.warning(f"获取嵌入向量时出错：{e}")
                if len(text) <= 1:
                    return None
                text = text[: int(len(text) * 0.9)]  # 缩短文本长度并重试
                await asyncio.sleep(0.07)
        return None

    async def calculate_resume_scores_for_collection(
        self,
        collection_name: str,
        query_contents: List[Dict[str, str]],
        field_relevance_scores: Dict[str, float],
        scoring_method: str = "hybrid",
        max_results: int = 100,
        top_similarities_count: int = 3,
        similarity_threshold: float = 0.5,
        decay_factor: float = 0.35,
    ) -> List[Tuple[str, float]]:
        """
        异步计算单个集合中简历的得分。

        Args:
            collection_name (str): Milvus 集合名称
            query_contents (List[Dict[str, str]]): 查询内容列表
            field_relevance_scores (Dict[str, float]): 字段相关性得分
            scoring_method (str): 评分方法，默认为'hybrid'
            max_results (int): 最大结果数，默认为100
            top_similarities_count (int): 考虑的最高相似度数量，默认为3
            similarity_threshold (float): 相似度阈值，默认为0.5
            decay_factor (float): 衰减因子，默认为0.35

        Returns:
            List[Tuple[str, float]]: 简历ID和得分的列表
        """
        resume_scores = {}

        for query in query_contents:
            field_name = query["field_name"]
            query_vector = await self.get_embedding(query["query_content"])
            if query_vector is None:
                logger.warning(f"无法为字段 {field_name} 获取嵌入向量，跳过此查询")
                continue

            num_entities = get_num_entities(collection_name)
            limit = clamp_search_limit(num_entities)

            logger.info(
                f"集合: {collection_name} 的实体数量: {num_entities} limit: {limit}"
            )

            # field_name 本身即最终向量列名（如 position_vector），按列名直接检索。
            # 返回的 distance 为相似度（1 - cosine_distance，越大越相似），与旧版内积口径一致。
            results = search_by_vector_column(
                collection_name,
                query_vector,
                field_name,
                top_k=limit,
                output_fields=["resume_id"],
            )

            for hit in results:
                resume_id = hit.get("resume_id")
                similarity = hit["distance"]

                if similarity < similarity_threshold:
                    continue

                if resume_id not in resume_scores:
                    resume_scores[resume_id] = {}
                if field_name not in resume_scores[resume_id]:
                    resume_scores[resume_id][field_name] = []
                resume_scores[resume_id][field_name].append(similarity)

        final_scores = {}
        for resume_id, field_scores in resume_scores.items():
            resume_score = 0
            for field_name, similarities in field_scores.items():
                top_similarities = sorted(similarities, reverse=True)[
                    :top_similarities_count
                ]

                if scoring_method == "sum":
                    field_score = sum(top_similarities)
                elif scoring_method == "max":
                    field_score = max(top_similarities)
                elif scoring_method == "hybrid":
                    field_score = max(top_similarities) + decay_factor * sum(
                        top_similarities[1:]
                    )
                else:
                    raise ValueError("无效的计算方法")

                resume_score += field_score * field_relevance_scores[field_name]

            final_scores[resume_id] = resume_score

        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:max_results]

    async def calculate_overall_resume_scores(
        self,
        refined_query: str,
        collection_relevances: List[Dict],
        collection_search_strategies: Dict,
        top_n: int = 3,
    ) -> pd.DataFrame:
        """
        异步计算所有集合的综合简历得分并返回前N个简历。

        Args:
            refined_query (str): 精炼后的查询
            collection_relevances (List[Dict]): 集合相关性列表
            collection_search_strategies (Dict): 集合搜索策略
            top_n (int): 要返回的最佳匹配简历数量

        Returns:
            pd.DataFrame: 包含排名后的简历得分的DataFrame
        """
        connect_to_milvus(db_name=self.db_name)

        all_scores = {}

        async def process_collection(collection_info):
            collection_name = collection_info["collection_name"]
            collection_weight = collection_info["relevance_score"]

            if collection_weight == 0:
                return

            collection_strategy = collection_search_strategies[collection_name]

            query_contents = []
            field_relevance_scores = {}
            for query in collection_strategy.vector_field_queries:
                query_contents.append(
                    {
                        "field_name": query.field_name,
                        "query_content": query.query_content,
                    }
                )
                # 同一向量字段可能对应多条 query（如 skills 最多 3 条，且策略要求所有
                # query 的 relevance 之和为 1）；相似度按 field_name 汇合打分，权重也须
                # 按 field_name 累加，否则只保留最后一条会把该字段权重少算数倍、压低排序。
                field_relevance_scores[query.field_name] = (
                    field_relevance_scores.get(query.field_name, 0) + query.relevance_score
                )

            collection_scores = await self.calculate_resume_scores_for_collection(
                collection_name=collection_name,
                query_contents=query_contents,
                field_relevance_scores=field_relevance_scores,
                scoring_method="hybrid",
                max_results=get_num_entities(collection_name),
                top_similarities_count=3,
                similarity_threshold=0.5,
                decay_factor=0.35,
            )

            return collection_name, collection_weight, collection_scores

        tasks = [process_collection(info) for info in collection_relevances]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                collection_name, collection_weight, collection_scores = result
                for resume_id, score in collection_scores:
                    if resume_id not in all_scores:
                        all_scores[resume_id] = {"total_score": 0}
                    all_scores[resume_id][collection_name] = score * collection_weight
                    all_scores[resume_id]["total_score"] += score * collection_weight

        df = pd.DataFrame.from_dict(all_scores, orient="index")
        df.index.name = "resume_id"
        df = df.reset_index()

        df = df.sort_values("total_score", ascending=False).head(top_n)

        for collection_info in collection_relevances:
            if collection_info["collection_name"] not in df.columns:
                df[collection_info["collection_name"]] = 0

        column_order = ["resume_id", "total_score"] + [
            info["collection_name"] for info in collection_relevances
        ]
        df = df[column_order]

        logger.info(f"已完成简历评分，正在筛选最佳匹配的 {top_n} 份简历")

        return df
