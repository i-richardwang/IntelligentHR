import os
import pandas as pd
import time
import numpy as np
from pymilvus import Collection, connections
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
import asyncio


class ResumeScorer:
    """简历评分器，负责计算简历的得分"""

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("EMBEDDING_API_KEY"),
        )
        self.connection_args = {
            "host": os.getenv("VECTOR_DB_HOST", "localhost"),
            "port": os.getenv("VECTOR_DB_PORT", "19530"),
            "db_name": os.getenv("VECTOR_DB_DATABASE_RESUME", "resume"),
        }

    async def get_embedding(self, text: str) -> List[float]:
        """
        异步获取文本的嵌入向量。

        Args:
            text (str): 输入文本

        Returns:
            List[float]: 嵌入向量，如果出现错误则返回 None
        """
        max_retries = 3
        for _ in range(max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=os.getenv("EMBEDDING_MODEL", ""), input=text
                )
                await asyncio.sleep(0.07)  # 避免频繁请求
                return response.data[0].embedding
            except Exception as e:
                print(f"获取嵌入向量时出错：{e}")
                if len(text) <= 1:
                    return None
                text = text[: int(len(text) * 0.9)]  # 缩短文本长度并重试
                await asyncio.sleep(0.07)
        return None

    async def calculate_resume_scores_for_collection(
        self,
        collection: Collection,
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
            collection (Collection): Milvus集合
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
                print(f"无法为字段 {field_name} 获取嵌入向量，跳过此查询")
                continue

            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            limit = max(collection.num_entities, 1)

            print(
                f"集合: {collection.name} 的实体数量: {collection.num_entities} limit: {limit}"
            )

            results = collection.search(
                data=[query_vector],
                anns_field=field_name,
                param=search_params,
                limit=limit,
                expr=None,
                output_fields=["resume_id"],
            )

            for hits in results:
                for hit in hits:
                    resume_id = hit.entity.get("resume_id")
                    similarity = hit.score

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
        connections.connect(
            host=self.connection_args["host"],
            port=self.connection_args["port"],
            db_name=self.connection_args["db_name"],
        )

        all_scores = {}

        async def process_collection(collection_info):
            collection_name = collection_info["collection_name"]
            collection_weight = collection_info["relevance_score"]

            if collection_weight == 0:
                return

            collection = Collection(collection_name)
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
                field_relevance_scores[query.field_name] = query.relevance_score

            collection_scores = await self.calculate_resume_scores_for_collection(
                collection=collection,
                query_contents=query_contents,
                field_relevance_scores=field_relevance_scores,
                scoring_method="hybrid",
                max_results=collection.num_entities,
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
        df.reset_index(inplace=True)

        df = df.sort_values("total_score", ascending=False).head(top_n)

        for collection_info in collection_relevances:
            if collection_info["collection_name"] not in df.columns:
                df[collection_info["collection_name"]] = 0

        column_order = ["resume_id", "total_score"] + [
            info["collection_name"] for info in collection_relevances
        ]
        df = df[column_order]

        print(f"已完成简历评分，正在筛选最佳匹配的 {top_n} 份简历")

        return df
