import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pymilvus import Collection, connections
from openai import OpenAI
from utils.dataset_utils import save_df_to_csv

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1", api_key=os.getenv("EMBEDDING_API_KEY")
)


def get_embedding(text: str) -> List[float]:
    """
    获取文本的嵌入向量。

    Args:
        text (str): 输入文本。

    Returns:
        List[float]: 嵌入向量。

    注意：
        如果文本长度过长导致API调用失败，会逐步缩短文本长度并重试。
    """
    while True:
        try:
            response = client.embeddings.create(
                model="BAAI/bge-large-en-v1.5", input=text
            )
            time.sleep(0.07)  # 避免超过API速率限制
            return response.data[0].embedding
        except Exception as e:
            if len(text) <= 1:
                return None
            text = text[: int(len(text) * 0.9)]
            time.sleep(0.07)


def calculate_resume_scores_for_collection(
    collection: Collection,
    query_contents: List[Dict[str, str]],
    field_relevance_scores: Dict[str, float],
    scoring_method: str = "hybrid",
    max_results: int = 100,
    top_similarities_count: int = 3,
    similarity_threshold: float = 0.3,
    decay_factor: float = 0.3,
) -> List[Tuple[str, float]]:
    """
    计算单个集合中简历的得分。

    Args:
        collection (Collection): Milvus集合对象。
        query_contents (List[Dict[str, str]]): 查询内容列表。
        field_relevance_scores (Dict[str, float]): 字段相关性得分。
        scoring_method (str): 评分方法，默认为'hybrid'。
        max_results (int): 最大结果数，默认为100。
        top_similarities_count (int): 考虑的最高相似度数量，默认为3。
        similarity_threshold (float): 相似度阈值，默认为0.3。
        decay_factor (float): 衰减因子，默认为0.3。

    Returns:
        List[Tuple[str, float]]: 简历ID和得分的列表。
    """
    resume_scores = {}

    for query in query_contents:
        field_name = query["field_name"]
        query_vector = query["query_content"]

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector],
            anns_field=field_name,
            param=search_params,
            limit=collection.num_entities,
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


def calculate_overall_resume_scores(state: Dict) -> Dict:
    """
    计算所有集合中简历的综合得分并返回前N个简历。

    Args:
        state (Dict): 当前状态字典。

    Returns:
        Dict: 更新后的状态字典，包含排序后的简历得分。

    注意：
        此函数会连接到Milvus数据库并执行查询操作。
    """
    connection_args = {"host": "localhost", "port": "19530", "db_name": "resume"}
    connections.connect(
        host=connection_args["host"],
        port=connection_args["port"],
        db_name=connection_args["db_name"],
    )

    all_scores = {}
    top_n = 3  # 可根据需求调整

    for collection_info in state["collection_relevances"]:
        collection_name = collection_info["collection_name"]
        collection_weight = collection_info["relevance_score"]

        if collection_weight == 0:
            continue

        collection = Collection(collection_name)
        collection_strategy = state["collection_search_strategies"][collection_name]

        query_contents = []
        field_relevance_scores = {}
        for query in collection_strategy["vector_field_queries"]:
            query_contents.append(
                {
                    "field_name": query["field_name"],
                    "query_content": get_embedding(query["query_content"]),
                }
            )
            field_relevance_scores[query["field_name"]] = query["relevance_score"]

        collection_scores = calculate_resume_scores_for_collection(
            collection=collection,
            query_contents=query_contents,
            field_relevance_scores=field_relevance_scores,
            scoring_method="hybrid",
            max_results=collection.num_entities,
            top_similarities_count=3,
            similarity_threshold=0.3,
            decay_factor=0.3,
        )

        for resume_id, score in collection_scores:
            if resume_id not in all_scores:
                all_scores[resume_id] = {"total_score": 0}
            all_scores[resume_id][collection_name] = score * collection_weight
            all_scores[resume_id]["total_score"] += score * collection_weight

    df = pd.DataFrame.from_dict(all_scores, orient="index")
    df.index.name = "resume_id"
    df.reset_index(inplace=True)

    df = df.sort_values("total_score", ascending=False).head(top_n)

    for collection_info in state["collection_relevances"]:
        if collection_info["collection_name"] not in df.columns:
            df[collection_info["collection_name"]] = 0

    column_order = ["resume_id", "total_score"] + [
        info["collection_name"] for info in state["collection_relevances"]
    ]
    df = df[column_order]

    state["ranked_resume_scores_file"] = save_df_to_csv(df, "ranked_resume_scores.csv")

    print("已完成简历评分，正在筛选最佳匹配的简历")

    return state