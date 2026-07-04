"""
Milvus 向量数据库工具（MilvusClient）。

pymilvus 3.0 起，ORM 风格接口（``connections`` / ``Collection`` / ``utility`` /
``FieldSchema`` / ``CollectionSchema``）已被官方标记为弃用并将在后续大版本移除，
统一改用 ``MilvusClient``。本模块据此重写，对外契约保持不变：

- ``connect_to_milvus(db_name)`` 建立/切换到目标数据库，缓存为模块级客户端单例；
- ``initialize_vector_store`` / ``create_milvus_collection`` 返回 ``collection_name``
  字符串（旧版返回 ORM ``Collection`` 对象，消费方多为"取回后原样透传给其它
  wrapper"，故改为字符串后调用链不变）；
- ``search_in_milvus`` / ``asearch_in_milvus`` 仍返回 ``List[Dict[str, Any]]``，
  键为非向量、非主键字段名加上 ``distance``，与旧版一致。

连接参数沿用 IHR 约定：``VECTOR_DB_HOST`` / ``VECTOR_DB_PORT``（可选
``VECTOR_DB_TOKEN`` 用于鉴权）。
"""

import os
import asyncio
import logging
from typing import List, Dict, Any

from pymilvus import MilvusClient, DataType

logger = logging.getLogger(__name__)

# 每个数据库一个 MilvusClient，进程内缓存复用。MilvusClient 设计为长生命周期
# 对象（底层连接自带池化），无需像旧 ORM 那样每次操作前 connect、操作后 disconnect。
_clients: Dict[str, MilvusClient] = {}
_current_db: str = "default"


def _build_uri() -> str:
    """按 IHR 环境变量约定拼出 Milvus 连接 URI。"""
    host = os.getenv("VECTOR_DB_HOST", "localhost")
    port = os.getenv("VECTOR_DB_PORT", "19530")
    return f"http://{host}:{port}"


def connect_to_milvus(db_name: str = "default") -> MilvusClient:
    """
    获取目标数据库的 MilvusClient（首次访问时创建并缓存），并置为当前库。

    Args:
        db_name (str): 目标数据库名。空字符串按 Milvus 语义回退到 ``default``。

    Returns:
        MilvusClient: 绑定到目标数据库的客户端。
    """
    global _current_db
    db = db_name or "default"
    if db not in _clients:
        _clients[db] = MilvusClient(
            uri=_build_uri(),
            token=os.getenv("VECTOR_DB_TOKEN", ""),
            db_name=db,
        )
    _current_db = db
    return _clients[db]


def get_milvus_client() -> MilvusClient:
    """返回当前数据库的 MilvusClient（尚未连接时以默认库建立）。"""
    return connect_to_milvus(_current_db)


def close_all() -> None:
    """关闭并清空所有缓存的 MilvusClient（供应用关闭时优雅清理）。"""
    for client in _clients.values():
        try:
            client.close()
        except Exception as e:  # noqa: BLE001 - 关闭失败不应阻断业务
            logger.warning("关闭 Milvus 连接时出错：%s", e)
    _clients.clear()


def _get_field_names(collection_name: str) -> List[str]:
    """返回集合的全部字段名。"""
    client = get_milvus_client()
    return [f["name"] for f in client.describe_collection(collection_name)["fields"]]


def get_field_names(collection_name: str) -> List[str]:
    """公开的字段名查询（供消费方替代旧版 ``collection.schema.fields``）。"""
    return _get_field_names(collection_name)


def _scalar_fields(field_names: List[str]) -> List[str]:
    """非主键、非向量的标量字段名。"""
    return [n for n in field_names if n != "id" and not n.endswith("_vector")]


def _vector_fields(field_names: List[str]) -> List[str]:
    """向量字段名（约定以 ``_vector`` 结尾）。"""
    return [n for n in field_names if n.endswith("_vector")]


def get_num_entities(collection_name: str) -> int:
    """返回集合实体数量（对应旧版 ``collection.num_entities``）。"""
    client = get_milvus_client()
    stats = client.get_collection_stats(collection_name)
    return int(stats.get("row_count", 0))


def initialize_vector_store(collection_name: str) -> str:
    """
    校验集合存在并加载，返回集合名。

    Args:
        collection_name (str): 集合名称。

    Returns:
        str: 集合名称（已加载）。

    Raises:
        ValueError: 如果集合不存在。
    """
    client = get_milvus_client()
    if not client.has_collection(collection_name):
        raise ValueError(
            f"Collection {collection_name} does not exist. Please create it first."
        )
    client.load_collection(collection_name)
    return collection_name


def create_milvus_collection(collection_config: Dict[str, Any], dim: int) -> str:
    """
    创建 Milvus 集合，支持多个向量字段，并为向量字段创建索引。

    Args:
        collection_config (Dict[str, Any]): 集合配置。
        dim (int): 向量维度。

    Returns:
        str: 创建并加载后的集合名称。
    """
    client = get_milvus_client()
    name = collection_config["name"]

    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    for field in collection_config["fields"]:
        schema.add_field(field["name"], DataType.VARCHAR, max_length=65535)
        if field.get("is_vector", False):
            schema.add_field(
                f"{field['name']}_vector", DataType.FLOAT_VECTOR, dim=dim
            )

    index_params = client.prepare_index_params()
    for field in collection_config["fields"]:
        if field.get("is_vector", False):
            index_params.add_index(
                field_name=f"{field['name']}_vector",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 1024},
            )

    client.create_collection(name, schema=schema, index_params=index_params)
    client.load_collection(name)
    return name


def _build_row(
    record: Dict[str, Any],
    vector: Dict[str, List[float]],
    scalar_fields: List[str],
    vector_fields: List[str],
) -> Dict[str, Any]:
    """按集合 schema 组装一行 MilvusClient 行式实体（scalar + vector）。"""
    entity: Dict[str, Any] = {sf: record.get(sf) for sf in scalar_fields}
    for vfield in vector_fields:
        original = vfield[: -len("_vector")]
        entity[vfield] = vector.get(original)
    return entity


def insert_to_milvus(
    collection_name: str,
    data: List[Dict[str, Any]],
    vectors: Dict[str, List[List[float]]],
) -> None:
    """
    将数据插入 Milvus 集合，支持多个向量字段。

    Args:
        collection_name (str): 集合名称。
        data (List[Dict[str, Any]]): 要插入的数据，每个字典代表一行标量字段。
        vectors (Dict[str, List[List[float]]]): 对应向量数据，键为原始字段名，
            值为与 ``data`` 行对齐的向量列表。
    """
    client = get_milvus_client()
    field_names = _get_field_names(collection_name)
    scalar_fields = _scalar_fields(field_names)
    vector_fields = _vector_fields(field_names)

    rows = [
        _build_row(
            record,
            {vf[: -len("_vector")]: vectors.get(vf[: -len("_vector")], [])[i]
             for vf in vector_fields},
            scalar_fields,
            vector_fields,
        )
        for i, record in enumerate(data)
    ]
    if rows:
        client.insert(collection_name, rows)


def update_milvus_records(
    collection_name: str,
    data: List[Dict[str, Any]],
    vectors: Dict[str, List[List[float]]],
    embedding_fields: List[str],
) -> None:
    """
    更新 Milvus 集合中的记录，支持多个向量字段。如果记录不存在，则插入新记录。

    Args:
        collection_name (str): 集合名称。
        data (List[Dict[str, Any]]): 要更新的数据，每个字典代表一行标量字段。
        vectors (Dict[str, List[List[float]]]): 对应向量数据，键为原始字段名。
        embedding_fields (List[str]): 用于定位既有记录的字段名列表。
    """
    client = get_milvus_client()
    field_names = _get_field_names(collection_name)
    scalar_fields = _scalar_fields(field_names)
    vector_fields = _vector_fields(field_names)

    for i, record in enumerate(data):
        # 用 embedding_fields 构建唯一定位表达式（值已在上游做过转义处理）
        query_expr = " and ".join(
            [f"{field} == '{record[field]}'" for field in embedding_fields]
        )
        existing = client.query(
            collection_name, filter=query_expr, output_fields=["id"]
        )
        if existing:
            client.delete(collection_name, ids=[r["id"] for r in existing])

        row = _build_row(
            record,
            {vf[: -len("_vector")]: vectors[vf[: -len("_vector")]][i]
             for vf in vector_fields},
            scalar_fields,
            vector_fields,
        )
        client.insert(collection_name, [row])


def delete_from_milvus(collection_name: str, filter_expr: str) -> None:
    """按布尔表达式删除记录（对应旧版 ``collection.delete(expr)``）。"""
    client = get_milvus_client()
    client.delete(collection_name, filter=filter_expr)


def _parse_hits(
    hits: List[Dict[str, Any]], output_fields: List[str]
) -> List[Dict[str, Any]]:
    """将 MilvusClient search 单路结果解析为与旧版一致的扁平 dict 列表。"""
    return [
        {
            **{field: hit["entity"].get(field) for field in output_fields},
            "distance": hit["distance"],
        }
        for hit in hits
    ]


def _search_output_fields(field_names: List[str]) -> List[str]:
    return [n for n in field_names if not n.endswith("_vector") and n != "id"]


def search_in_milvus(
    collection_name: str,
    query_vector: List[float],
    vector_field: str,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    在 Milvus 集合中搜索最相似的向量。

    Args:
        collection_name (str): 集合名称。
        query_vector (List[float]): 查询向量。
        vector_field (str): 目标向量字段的原始名（内部自动加 ``_vector`` 后缀）。
        top_k (int): 返回的最相似结果数量。默认为 1。

    Returns:
        List[Dict[str, Any]]: 搜索结果列表，键为标量字段名加 ``distance``。
    """
    client = get_milvus_client()
    output_fields = _search_output_fields(_get_field_names(collection_name))

    results = client.search(
        collection_name,
        data=[query_vector],
        anns_field=f"{vector_field}_vector",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=output_fields,
    )
    return _parse_hits(results[0], output_fields)


async def asearch_in_milvus(
    collection_name: str,
    query_vector: List[float],
    vector_field: str,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """``search_in_milvus`` 的异步封装（在线程中执行同步调用）。"""
    return await asyncio.to_thread(
        search_in_milvus, collection_name, query_vector, vector_field, top_k
    )
