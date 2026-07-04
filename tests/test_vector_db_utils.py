"""``utils.vector_db_utils`` 的单元测试（MilvusClient 迁移回归保护）。

pymilvus 3.0 起 ORM 接口弃用，本模块已改用 MilvusClient。这些测试用 mock 客户端
锁定迁移中最易错的行为差异，无需真实 Milvus 服务端：

- ``search`` 返回 ``List[List[dict]]``（hit 为 ``{"id","distance","entity"}``），
  解析后须扁平为 ``{标量字段..., "distance"}``；
- ``insert`` 是行式 ``list[dict]``（旧 ORM 是列式 list-of-lists）；
- 向量字段按 ``{field}_vector`` 命名，anns_field 自动加后缀、output_fields 排除
  主键与向量字段；
- ``query`` / ``delete`` 用 ``filter=`` 表达式，``num_entities`` 来自
  ``get_collection_stats()["row_count"]``。
"""

from unittest.mock import MagicMock

import pytest

import utils.vector_db_utils as vdb


# 典型集合 schema：id(主键) + 4 个标量字段 + 1 个向量字段
FIELDS = [
    {"name": "id"},
    {"name": "resume_id"},
    {"name": "raw_text"},
    {"name": "file_name"},
    {"name": "upload_date"},
    {"name": "raw_text_vector"},
]


@pytest.fixture
def client(monkeypatch):
    """返回一个 mock MilvusClient，并让模块内部一律取用它。"""
    c = MagicMock()
    c.describe_collection.return_value = {"fields": FIELDS}
    c.has_collection.return_value = True
    c.get_collection_stats.return_value = {"row_count": 42}
    monkeypatch.setattr(vdb, "get_milvus_client", lambda: c)
    return c


def test_get_field_names(client):
    assert vdb.get_field_names("c") == [
        "id",
        "resume_id",
        "raw_text",
        "file_name",
        "upload_date",
        "raw_text_vector",
    ]


def test_get_num_entities_reads_row_count(client):
    assert vdb.get_num_entities("c") == 42


def test_initialize_vector_store_returns_name_and_loads(client):
    assert vdb.initialize_vector_store("c") == "c"
    client.load_collection.assert_called_once_with("c")


def test_initialize_vector_store_missing_raises(client):
    client.has_collection.return_value = False
    with pytest.raises(ValueError):
        vdb.initialize_vector_store("missing")


def test_search_parses_hits_and_suffixes_anns_field(client):
    client.search.return_value = [
        [
            {
                "id": 1,
                "distance": 0.95,
                "entity": {
                    "resume_id": "r1",
                    "raw_text": "hi",
                    "file_name": "a.pdf",
                    "upload_date": "2026-07-04",
                },
            }
        ]
    ]

    out = vdb.search_in_milvus("c", [0.1, 0.2], "raw_text", top_k=5)

    # 返回契约：扁平 dict，键为标量字段 + distance
    assert out == [
        {
            "resume_id": "r1",
            "raw_text": "hi",
            "file_name": "a.pdf",
            "upload_date": "2026-07-04",
            "distance": 0.95,
        }
    ]

    kwargs = client.search.call_args.kwargs
    assert kwargs["anns_field"] == "raw_text_vector"  # 自动加 _vector 后缀
    assert kwargs["limit"] == 5
    # output_fields 排除主键与向量字段
    assert "id" not in kwargs["output_fields"]
    assert "raw_text_vector" not in kwargs["output_fields"]
    assert set(kwargs["output_fields"]) == {
        "resume_id",
        "raw_text",
        "file_name",
        "upload_date",
    }


def test_insert_builds_row_based_entities(client):
    data = [
        {
            "resume_id": "r1",
            "raw_text": "hello",
            "file_name": "a.pdf",
            "upload_date": "2026-07-04",
        }
    ]
    vectors = {"raw_text": [[0.1, 0.2, 0.3]]}

    vdb.insert_to_milvus("c", data, vectors)

    name, rows = client.insert.call_args.args
    assert name == "c"
    # 行式 dict：标量字段 + {field}_vector
    assert rows == [
        {
            "resume_id": "r1",
            "raw_text": "hello",
            "file_name": "a.pdf",
            "upload_date": "2026-07-04",
            "raw_text_vector": [0.1, 0.2, 0.3],
        }
    ]


def test_update_deletes_existing_then_inserts(client):
    client.query.return_value = [{"id": 7}]
    data = [
        {
            "resume_id": "r1",
            "raw_text": "hello",
            "file_name": "a.pdf",
            "upload_date": "2026-07-04",
        }
    ]
    vectors = {"raw_text": [[0.4, 0.5, 0.6]]}

    vdb.update_milvus_records("c", data, vectors, embedding_fields=["resume_id"])

    # 命中既有记录 → 先按 id 删除，再插入
    client.delete.assert_called_once_with("c", ids=[7])
    _, rows = client.insert.call_args.args
    assert rows[0]["raw_text_vector"] == [0.4, 0.5, 0.6]
    # query 用 filter 表达式定位
    assert "resume_id == 'r1'" in client.query.call_args.kwargs["filter"]


def test_update_inserts_when_no_existing(client):
    client.query.return_value = []
    data = [{"resume_id": "r1", "raw_text": "x", "file_name": "f", "upload_date": "d"}]
    vectors = {"raw_text": [[0.1]]}

    vdb.update_milvus_records("c", data, vectors, embedding_fields=["resume_id"])

    client.delete.assert_not_called()
    client.insert.assert_called_once()


def test_delete_from_milvus_passes_filter(client):
    vdb.delete_from_milvus("c", 'resume_id == "r1"')
    client.delete.assert_called_once_with("c", filter='resume_id == "r1"')
