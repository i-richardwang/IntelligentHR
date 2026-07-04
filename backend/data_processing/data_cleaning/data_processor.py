import os
from typing import List, Dict, Callable

from utils.llm_tools import create_embeddings
from utils.vector_db_utils import (
    connect_to_milvus,
    initialize_vector_store as init_store,
    asearch_in_milvus,
    get_field_names,
)


def initialize_vector_store(collection_name: str) -> str:
    """初始化或加载向量存储"""
    connect_to_milvus(os.getenv("VECTOR_DB_DATA_CLEANING", "default"))
    return init_store(collection_name)


def get_entity_retriever(collection: str, entity_type: str) -> Callable:
    """获取实体检索器"""

    async def retriever(query: str, k: int = 1) -> List[Dict]:
        embedding_model = create_embeddings()
        query_embedding = await embedding_model.aembed_query(query)

        # 获取集合的字段信息
        field_names = get_field_names(collection)

        # 获取原始名称字段
        original_name_field = (
            "company_name" if "company_name" in field_names else "school_name"
        )

        # 在搜索时使用原始名称字段
        results = await asearch_in_milvus(collection, query_embedding, original_name_field, k)

        # 确定原始名称字段和标准名称字段
        standard_name_field = (
            "standard_name" if "standard_name" in field_names else original_name_field
        )

        retrieved_entities = []
        for result in results:
            entity = {
                "original_name": result.get(original_name_field),
                "standard_name": result.get(standard_name_field),
                "distance": result.get("distance"),
            }
            retrieved_entities.append(entity)

        return retrieved_entities

    return retriever
