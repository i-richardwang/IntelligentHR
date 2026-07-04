"""``utils.vector_db_utils`` 的集成测试（libSQL 迁移回归保护）。

向量层已从 Milvus（需独立 server）迁到 libSQL —— SQLite 的开源分支，向量能力进内核，
可在进程内直接建库。故这些测试不再 mock，而是对真实 libSQL 临时库跑通全链路，锁定
迁移关键不变量：

- 单表多向量：``{field}_vector F32_BLOB`` 列，各字段独立检索；
- 度量对齐：入库/查询两侧归一化后 ``distance == 归一化点积(IP)``，且"越大越相似"，
  沿用旧版内积语义与阈值判定；
- 两套字段命名：``search_in_milvus`` 补 ``_vector`` 后缀，``search_by_vector_column``
  直用最终列名；
- ``update`` 先删后插（按 embedding_fields 定位），``delete`` 按字段等值删除。
"""

import math

import pytest

import utils.vector_db_utils as vdb


CONFIG = {
    "name": "work_experiences",
    "fields": [
        {"name": "resume_id"},
        {"name": "company"},
        {"name": "position", "is_vector": True},
        {"name": "responsibilities", "is_vector": True},
    ],
    "embedding_fields": ["position", "responsibilities"],
}


def _norm_ip(a, b):
    """归一化点积（IP），用于对照 libSQL 返回的 distance。"""
    def n(v):
        s = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / s for x in v]
    na, nb = n(a), n(b)
    return sum(x * y for x, y in zip(na, nb))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """把向量库指向临时目录，清空连接缓存，建好一个双向量集合。"""
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path))
    vdb.close_all()  # 清掉可能残留的、指向旧路径的缓存连接
    vdb.connect_to_milvus("testdb")
    name = vdb.create_milvus_collection(CONFIG, dim=4)
    yield name
    vdb.close_all()


def _seed(name):
    data = [
        {"resume_id": "r1", "company": "Acme", "position": "eng", "responsibilities": "build"},
        {"resume_id": "r2", "company": "Globex", "position": "mgr", "responsibilities": "lead"},
    ]
    vectors = {
        "position": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "responsibilities": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    }
    vdb.insert_to_milvus(name, data, vectors)


def test_create_collection_has_scalar_and_vector_columns(store):
    fields = vdb.get_field_names(store)
    assert fields == [
        "id",
        "resume_id",
        "company",
        "position",
        "position_vector",
        "responsibilities",
        "responsibilities_vector",
    ]


def test_insert_and_num_entities(store):
    _seed(store)
    assert vdb.get_num_entities(store) == 2


def test_search_in_milvus_suffixes_field_and_returns_scalars(store):
    _seed(store)
    q = [0.9, 0.1, 0.0, 0.0]
    out = vdb.search_in_milvus(store, q, "position", top_k=2)

    # 最相似为 r1，返回标量字段 + distance
    assert out[0]["resume_id"] == "r1"
    assert set(out[0].keys()) == {
        "resume_id",
        "company",
        "position",
        "responsibilities",
        "distance",
    }
    # distance 为相似度（越大越相似），且等于归一化点积
    assert out[0]["distance"] >= out[1]["distance"]
    assert out[0]["distance"] == pytest.approx(_norm_ip(q, [1.0, 0.0, 0.0, 0.0]), abs=1e-6)


def test_search_by_vector_column_uses_final_column_name(store):
    _seed(store)
    q = [0.0, 0.0, 0.9, 0.1]
    # 直接按最终列名（含 _vector）检索，output_fields 限定
    out = vdb.search_by_vector_column(
        store, q, "responsibilities_vector", top_k=2, output_fields=["resume_id"]
    )
    assert out[0]["resume_id"] == "r1"  # r1 的 responsibilities 向量是 [0,0,1,0]
    assert set(out[0].keys()) == {"resume_id", "distance"}


def test_update_replaces_in_place(store):
    _seed(store)
    vdb.update_milvus_records(
        store,
        [{"resume_id": "r1", "company": "Acme2", "position": "x", "responsibilities": "y"}],
        {"position": [[0.0, 0.0, 0.0, 1.0]], "responsibilities": [[0.0, 0.0, 1.0, 0.0]]},
        embedding_fields=["resume_id"],
    )
    # 实体数不变（替换而非新增）
    assert vdb.get_num_entities(store) == 2
    out = vdb.search_in_milvus(store, [0.0, 0.0, 0.0, 1.0], "position", top_k=1)
    assert out[0]["resume_id"] == "r1"
    assert out[0]["company"] == "Acme2"


def test_update_inserts_when_no_existing(store):
    vdb.update_milvus_records(
        store,
        [{"resume_id": "r9", "company": "New", "position": "p", "responsibilities": "r"}],
        {"position": [[1.0, 0.0, 0.0, 0.0]], "responsibilities": [[0.0, 1.0, 0.0, 0.0]]},
        embedding_fields=["resume_id"],
    )
    assert vdb.get_num_entities(store) == 1


def test_delete_by_field(store):
    _seed(store)
    vdb.delete_from_milvus(store, "resume_id", "r2")
    assert vdb.get_num_entities(store) == 1
    remaining = vdb.get_all_records(store, output_fields=["resume_id"])
    assert [r["resume_id"] for r in remaining] == ["r1"]


def test_initialize_vector_store_returns_name(store):
    assert vdb.initialize_vector_store(store) == store


def test_initialize_vector_store_missing_raises(store):
    with pytest.raises(ValueError):
        vdb.initialize_vector_store("does_not_exist")


def test_has_collection(store):
    assert vdb.has_collection(store) is True
    assert vdb.has_collection("nope") is False


def test_num_entities_zero_for_missing_collection(store):
    assert vdb.get_num_entities("missing") == 0
