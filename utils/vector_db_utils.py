"""
向量存储工具（libSQL）。

本模块把项目的向量检索层从 Milvus（需独立 server）迁移到 libSQL —— SQLite 的开源
分支，向量能力做进了内核（``F32_BLOB`` 列类型 + ``vector_distance_cos`` 等标量函数），
纯本地单文件、零 server、离线可跑。对外契约保持不变，消费方无需感知底层更换：

- ``connect_to_milvus(db_name)`` 建立/切换到目标数据库（每个 db_name 对应一个本地
  ``.db`` 文件），缓存为模块级连接；
- ``initialize_vector_store`` / ``create_milvus_collection`` 返回 collection_name 字符串；
- ``search_in_milvus`` / ``asearch_in_milvus`` 仍返回 ``List[Dict[str, Any]]``，键为
  标量字段名加 ``distance``，且 ``distance`` 仍是"越大越相似"的相似度（见下）。

**度量对齐（关键）**：旧实现用 Milvus 的内积（IP）度量，消费方普遍以
``result["distance"] >= threshold`` 判定。libSQL 原生只有 cosine/L2。为精确复现 IP
语义，本模块在**入库与查询两侧都对向量做 L2 归一化**——归一化后
``1 - vector_distance_cos == 点积(IP)``（实测误差 ~1e-8），故对外返回的 ``distance``
统一取 ``1 - cosine_distance``，与旧版内积相似度逐位对齐，阈值/排序语义不变。

**字段命名两套约定（沿用旧版）**：
- ``search_in_milvus(name, qvec, field, k)`` 的 ``field`` 是**裸字段名**（如
  ``"description"``），内部自动加 ``_vector`` 后缀定位向量列；
- ``search_by_vector_column(name, qvec, column, ...)`` 的 ``column`` 是**最终列名**
  （如 ``"position_vector"``），供简历评分器直接按列名检索。

存储位置：环境变量 ``VECTOR_DB_PATH``（默认 ``data/vector_store``），每个逻辑数据库
落一个 ``{VECTOR_DB_PATH}/{db_name}.db`` 文件。
"""

import os
import re
import math
import asyncio
import logging
import threading
from typing import List, Dict, Any, Optional

import libsql

logger = logging.getLogger(__name__)

# 每个逻辑数据库一个 libSQL 连接，进程内缓存复用。
_connections: Dict[str, "libsql.Connection"] = {}
_current_db: str = "default"
# libSQL/SQLite 连接非线程安全；异步包装（asearch_in_milvus 经线程池）可能并发访问，
# 统一用可重入锁串行化。demo 规模下串行完全够用。
_lock = threading.RLock()

# 合法标识符（表名/列名无法参数化，只能拼接，故先校验白名单字符再使用）。
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _base_dir() -> str:
    return os.getenv("VECTOR_DB_PATH", "data/vector_store")


def _db_path(db_name: str) -> str:
    base = _base_dir()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{db_name}.db")


def _ident(name: str) -> str:
    """校验并返回可安全拼接进 SQL 的标识符。"""
    if not _IDENT_RE.match(name):
        raise ValueError(f"非法标识符：{name!r}")
    return name


def _normalize(vec: List[float]) -> List[float]:
    """L2 归一化：归一化后 cosine 距离与内积严格对应（零向量原样返回）。"""
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm == 0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def _vec_literal(vec: List[float]) -> str:
    """序列化为 libSQL ``vector32(?)`` 接受的 JSON 字符串。"""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def connect_to_milvus(db_name: str = "default") -> "libsql.Connection":
    """
    获取目标数据库的 libSQL 连接（首次访问时创建并缓存），并置为当前库。

    Args:
        db_name (str): 目标数据库名。空字符串回退到 ``default``。

    Returns:
        libsql.Connection: 绑定到目标 ``.db`` 文件的连接。
    """
    global _current_db
    db = db_name or "default"
    _ident(db)
    with _lock:
        if db not in _connections:
            _connections[db] = libsql.connect(_db_path(db))
        _current_db = db
        return _connections[db]


def get_milvus_client() -> "libsql.Connection":
    """返回当前数据库的 libSQL 连接（尚未连接时以默认库建立）。"""
    return connect_to_milvus(_current_db)


def close_all() -> None:
    """关闭并清空所有缓存连接（供应用关闭时优雅清理）。"""
    with _lock:
        for conn in _connections.values():
            try:
                conn.close()
            except Exception as e:  # noqa: BLE001 - 关闭失败不应阻断业务
                logger.warning("关闭 libSQL 连接时出错：%s", e)
        _connections.clear()


def _get_field_names(collection_name: str) -> List[str]:
    """返回集合（表）的全部列名。"""
    _ident(collection_name)
    conn = get_milvus_client()
    with _lock:
        cur = conn.execute(f"PRAGMA table_info({collection_name})")
        return [row[1] for row in cur.fetchall()]


def get_field_names(collection_name: str) -> List[str]:
    """公开的字段名查询（供消费方替代旧版 ``collection.schema.fields``）。"""
    return _get_field_names(collection_name)


def _scalar_fields(field_names: List[str]) -> List[str]:
    """非主键、非向量的标量字段名。"""
    return [n for n in field_names if n != "id" and not n.endswith("_vector")]


def _vector_fields(field_names: List[str]) -> List[str]:
    """向量字段名（约定以 ``_vector`` 结尾）。"""
    return [n for n in field_names if n.endswith("_vector")]


def _search_output_fields(field_names: List[str]) -> List[str]:
    """检索默认返回的字段：标量字段（排除主键与向量列）。"""
    return _scalar_fields(field_names)


def _has_collection(collection_name: str) -> bool:
    _ident(collection_name)
    conn = get_milvus_client()
    with _lock:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (collection_name,),
        ).fetchone()
    return row is not None


def has_collection(collection_name: str) -> bool:
    """集合（表）是否存在（供管理页替代旧版 ``utility.has_collection``）。"""
    return _has_collection(collection_name)


def get_all_records(
    collection_name: str, output_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """取回集合全部记录的指定标量字段（替代旧版 ``collection.query(expr="id>=0", ...)``）。"""
    field_names = _get_field_names(collection_name)
    fields = output_fields if output_fields is not None else _scalar_fields(field_names)
    safe = [_ident(f) for f in fields]
    if not safe:
        return []
    conn = get_milvus_client()
    with _lock:
        cur = conn.execute(f"SELECT {', '.join(safe)} FROM {_ident(collection_name)}")
        col_names = [c[0] for c in cur.description]
        rows = cur.fetchall()
    return [dict(zip(col_names, row)) for row in rows]


def get_num_entities(collection_name: str) -> int:
    """返回集合实体数量（对应旧版 ``collection.num_entities``）。"""
    if not _has_collection(collection_name):
        return 0
    _ident(collection_name)
    conn = get_milvus_client()
    with _lock:
        return int(conn.execute(f"SELECT COUNT(*) FROM {collection_name}").fetchone()[0])


def initialize_vector_store(collection_name: str) -> str:
    """
    校验集合存在，返回集合名。

    Args:
        collection_name (str): 集合名称。

    Returns:
        str: 集合名称。

    Raises:
        ValueError: 如果集合不存在。
    """
    if not _has_collection(collection_name):
        raise ValueError(
            f"Collection {collection_name} does not exist. Please create it first."
        )
    return collection_name


def create_milvus_collection(collection_config: Dict[str, Any], dim: int) -> str:
    """
    创建向量集合（表），支持多个向量字段。

    每个配置字段建为 ``TEXT`` 列；标记 ``is_vector`` 的字段额外建一个
    ``{field}_vector F32_BLOB(dim)`` 列存放归一化向量。

    Args:
        collection_config (Dict[str, Any]): 集合配置。
        dim (int): 向量维度。

    Returns:
        str: 创建后的集合名称。
    """
    name = _ident(collection_config["name"])
    conn = get_milvus_client()

    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
    for field in collection_config["fields"]:
        fname = _ident(field["name"])
        columns.append(f"{fname} TEXT")
        if field.get("is_vector", False):
            columns.append(f"{fname}_vector F32_BLOB({int(dim)})")

    ddl = f"CREATE TABLE IF NOT EXISTS {name} (\n  " + ",\n  ".join(columns) + "\n)"
    with _lock:
        conn.execute(ddl)
        conn.commit()
    return name


def _insert_row(
    conn: "libsql.Connection",
    collection_name: str,
    scalar_values: Dict[str, Any],
    vector_values: Dict[str, List[float]],
) -> None:
    """组装并执行一行插入（标量列直接绑定，向量列走 ``vector32(?)``，向量先归一化）。"""
    cols: List[str] = []
    placeholders: List[str] = []
    params: List[Any] = []

    for col, val in scalar_values.items():
        cols.append(_ident(col))
        placeholders.append("?")
        params.append(val)

    for col, vec in vector_values.items():
        cols.append(_ident(col))
        placeholders.append("vector32(?)")
        params.append(_vec_literal(_normalize(vec)))

    sql = (
        f"INSERT INTO {_ident(collection_name)} ({', '.join(cols)}) "
        f"VALUES ({', '.join(placeholders)})"
    )
    conn.execute(sql, tuple(params))


def _split_record(
    record: Dict[str, Any],
    vectors_for_row: Dict[str, List[float]],
    scalar_fields: List[str],
    vector_fields: List[str],
) -> tuple:
    """把一条记录拆成"标量列 -> 值"与"向量列 -> 向量"两个字典。"""
    scalar_values = {sf: record.get(sf) for sf in scalar_fields if sf in record}
    vector_values: Dict[str, List[float]] = {}
    for vfield in vector_fields:
        original = vfield[: -len("_vector")]
        if original in vectors_for_row:
            vector_values[vfield] = vectors_for_row[original]
    return scalar_values, vector_values


def insert_to_milvus(
    collection_name: str,
    data: List[Dict[str, Any]],
    vectors: Dict[str, List[List[float]]],
) -> None:
    """
    将数据插入向量集合，支持多个向量字段。

    Args:
        collection_name (str): 集合名称。
        data (List[Dict[str, Any]]): 每个字典代表一行标量字段。
        vectors (Dict[str, List[List[float]]]): 向量数据，键为原始字段名，
            值为与 ``data`` 行对齐的向量列表。
    """
    conn = get_milvus_client()
    field_names = _get_field_names(collection_name)
    scalar_fields = _scalar_fields(field_names)
    vector_fields = _vector_fields(field_names)

    with _lock:
        for i, record in enumerate(data):
            vectors_for_row = {
                vf[: -len("_vector")]: vectors[vf[: -len("_vector")]][i]
                for vf in vector_fields
                if vf[: -len("_vector")] in vectors
            }
            scalar_values, vector_values = _split_record(
                record, vectors_for_row, scalar_fields, vector_fields
            )
            _insert_row(conn, collection_name, scalar_values, vector_values)
        conn.commit()


def update_milvus_records(
    collection_name: str,
    data: List[Dict[str, Any]],
    vectors: Dict[str, List[List[float]]],
    embedding_fields: List[str],
) -> None:
    """
    更新集合记录，支持多个向量字段。命中既有记录先删后插，否则直接插入。

    Args:
        collection_name (str): 集合名称。
        data (List[Dict[str, Any]]): 每个字典代表一行标量字段。
        vectors (Dict[str, List[List[float]]]): 向量数据，键为原始字段名。
        embedding_fields (List[str]): 用于定位既有记录的字段名列表。
    """
    name = _ident(collection_name)
    conn = get_milvus_client()
    field_names = _get_field_names(collection_name)
    scalar_fields = _scalar_fields(field_names)
    vector_fields = _vector_fields(field_names)

    with _lock:
        for i, record in enumerate(data):
            # 用 embedding_fields 定位既有记录（参数化，杜绝注入）
            where = " AND ".join(f"{_ident(f)} = ?" for f in embedding_fields)
            where_params = tuple(record[f] for f in embedding_fields)
            existing = conn.execute(
                f"SELECT id FROM {name} WHERE {where}", where_params
            ).fetchall()
            if existing:
                ids = [r[0] for r in existing]
                q = ", ".join("?" for _ in ids)
                conn.execute(f"DELETE FROM {name} WHERE id IN ({q})", tuple(ids))

            vectors_for_row = {
                vf[: -len("_vector")]: vectors[vf[: -len("_vector")]][i]
                for vf in vector_fields
                if vf[: -len("_vector")] in vectors
            }
            scalar_values, vector_values = _split_record(
                record, vectors_for_row, scalar_fields, vector_fields
            )
            _insert_row(conn, name, scalar_values, vector_values)
        conn.commit()


def delete_from_milvus(collection_name: str, field: str, value: Any) -> None:
    """按单字段等值条件删除记录（参数化，对应旧版按表达式删除）。

    Args:
        collection_name (str): 集合名称。
        field (str): 用于匹配的字段名。
        value (Any): 匹配值。
    """
    name = _ident(collection_name)
    fld = _ident(field)
    conn = get_milvus_client()
    with _lock:
        conn.execute(f"DELETE FROM {name} WHERE {fld} = ?", (value,))
        conn.commit()


def _knn(
    collection_name: str,
    query_vector: List[float],
    vector_column: str,
    top_k: int,
    output_fields: List[str],
) -> List[Dict[str, Any]]:
    """暴力 KNN：``1 - vector_distance_cos`` 作为相似度（越大越相似），降序取前 k。"""
    name = _ident(collection_name)
    col = _ident(vector_column)
    safe_out = [_ident(f) for f in output_fields]

    select_cols = ", ".join(safe_out) if safe_out else "id"
    conn = get_milvus_client()
    qlit = _vec_literal(_normalize(query_vector))

    sql = (
        f"SELECT {select_cols}, "
        f"(1 - vector_distance_cos({col}, vector32(?))) AS distance "
        f"FROM {name} WHERE {col} IS NOT NULL "
        f"ORDER BY distance DESC LIMIT ?"
    )
    with _lock:
        cur = conn.execute(sql, (qlit, int(max(top_k, 1))))
        col_names = [c[0] for c in cur.description]
        rows = cur.fetchall()

    return [dict(zip(col_names, row)) for row in rows]


def search_in_milvus(
    collection_name: str,
    query_vector: List[float],
    vector_field: str,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    在集合中检索最相似的向量。

    Args:
        collection_name (str): 集合名称。
        query_vector (List[float]): 查询向量。
        vector_field (str): 目标向量字段的**原始名**（内部自动加 ``_vector`` 后缀）。
        top_k (int): 返回的最相似结果数量。默认为 1。

    Returns:
        List[Dict[str, Any]]: 结果列表，键为标量字段名加 ``distance``（相似度，越大越相似）。
    """
    output_fields = _search_output_fields(_get_field_names(collection_name))
    return _knn(
        collection_name, query_vector, f"{vector_field}_vector", top_k, output_fields
    )


def search_by_vector_column(
    collection_name: str,
    query_vector: List[float],
    vector_column: str,
    top_k: int = 1,
    output_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    按**最终列名**检索（供简历评分器：其 field_name 本身即 ``*_vector`` 列名）。

    Args:
        collection_name (str): 集合名称。
        query_vector (List[float]): 查询向量。
        vector_column (str): 向量列名（已含 ``_vector`` 后缀，原样使用）。
        top_k (int): 返回数量。
        output_fields (Optional[List[str]]): 需返回的标量字段；缺省返回全部标量字段。

    Returns:
        List[Dict[str, Any]]: 结果列表，键为 output_fields 加 ``distance``。
    """
    if output_fields is None:
        output_fields = _search_output_fields(_get_field_names(collection_name))
    return _knn(collection_name, query_vector, vector_column, top_k, output_fields)


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
