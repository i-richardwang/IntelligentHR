"""``utils.llm_tools`` 的 embedding 工厂与 ``CustomEmbeddings`` 客户端单元测试。

回归保护：
- ``create_embeddings`` 工厂统一从环境变量读取配置，缺失时快速失败（中文报错）；
  这消除了各调用点重复构造，以及某处硬编码供应商 base_url、换供应商换不掉的隐患。
- ``CustomEmbeddings`` 基于 httpx，提供同步与原生异步实现，二者请求体 / 头部 /
  端点一致，响应解析取 ``data[0].embedding``。
"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from utils.llm_tools import CustomEmbeddings, VectorEncoder, create_embeddings


EMBED_ENV = {
    "EMBEDDING_API_KEY": "test-key",
    "EMBEDDING_API_BASE": "https://api.example.com/v1/embeddings",
    "EMBEDDING_MODEL": "BAAI/bge-m3",
}


def _set_env(monkeypatch, **overrides):
    """设置 embedding 相关环境变量；值为 None 表示删除该变量。"""
    for name, value in {**EMBED_ENV, **overrides}.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _fake_response(embedding):
    resp = MagicMock()
    resp.raise_for_status.return_value = resp
    resp.json.return_value = {"data": [{"embedding": embedding}]}
    return resp


# ---------------- 工厂 ----------------


def test_factory_reads_env(monkeypatch):
    _set_env(monkeypatch)
    emb = create_embeddings()
    assert isinstance(emb, CustomEmbeddings)
    assert emb.api_key == "test-key"
    assert emb.api_url == "https://api.example.com/v1/embeddings"
    assert emb.model == "BAAI/bge-m3"


def test_factory_model_override(monkeypatch):
    _set_env(monkeypatch)
    emb = create_embeddings(model="custom-model")
    assert emb.model == "custom-model"


@pytest.mark.parametrize(
    "missing", ["EMBEDDING_API_KEY", "EMBEDDING_API_BASE", "EMBEDDING_MODEL"]
)
def test_factory_missing_config_raises(monkeypatch, missing):
    _set_env(monkeypatch, **{missing: None})
    with pytest.raises(ValueError) as exc:
        create_embeddings()
    # 报错信息应指明缺失的具体变量名
    assert missing in str(exc.value)


# ---------------- 同步 ----------------


def test_embed_query_sync():
    emb = CustomEmbeddings(api_key="k", api_url="http://x/embeddings", model="m")
    client = MagicMock()
    client.post.return_value = _fake_response([0.1, 0.2, 0.3])
    ctx = MagicMock()
    ctx.__enter__.return_value = client
    ctx.__exit__.return_value = False
    with patch("utils.llm_tools.httpx.Client", return_value=ctx):
        vec = emb.embed_query("你好")
    assert vec == [0.1, 0.2, 0.3]
    args, kwargs = client.post.call_args
    assert args[0] == "http://x/embeddings"
    assert kwargs["json"] == {"model": "m", "input": "你好", "encoding_format": "float"}
    assert kwargs["headers"]["authorization"] == "Bearer k"


def test_embed_documents_sync():
    emb = CustomEmbeddings(api_key="k", api_url="http://x/embeddings", model="m")
    client = MagicMock()
    client.post.side_effect = [_fake_response([1.0]), _fake_response([2.0])]
    ctx = MagicMock()
    ctx.__enter__.return_value = client
    ctx.__exit__.return_value = False
    with patch("utils.llm_tools.httpx.Client", return_value=ctx):
        vecs = emb.embed_documents(["a", "b"])
    assert vecs == [[1.0], [2.0]]
    assert client.post.call_count == 2


# ---------------- 原生异步 ----------------


def test_aembed_query_native():
    emb = CustomEmbeddings(api_key="k", api_url="http://x/embeddings", model="m")
    client = MagicMock()
    client.post = AsyncMock(return_value=_fake_response([0.4, 0.5]))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    with patch("utils.llm_tools.httpx.AsyncClient", return_value=ctx):
        vec = asyncio.run(emb.aembed_query("hi"))
    assert vec == [0.4, 0.5]
    client.post.assert_awaited_once()


# ---------------- 入库/查询 embedding 口径统一 ----------------
#
# 回归保护：入库侧曾硬编码 VectorEncoder(model="BAAI/bge-m3")，而查询侧走
# create_embeddings()/EMBEDDING_MODEL。两侧模型不同则向量不可比、相似度失真。
# 现入库侧改为 VectorEncoder()（model 缺省 -> EMBEDDING_MODEL），与查询侧同源。


def test_vector_encoder_defaults_to_env_model(monkeypatch):
    _set_env(monkeypatch, EMBEDDING_MODEL="env-model")
    # 缺省 model -> 经工厂读取 EMBEDDING_MODEL
    assert VectorEncoder().embeddings.model == "env-model"


def test_ingestion_and_query_use_same_model(monkeypatch):
    _set_env(monkeypatch, EMBEDDING_MODEL="single-source-model")
    ingestion_model = VectorEncoder().embeddings.model  # 入库侧
    query_model = create_embeddings().model  # 查询侧
    # 单一真源：两侧解析到同一 embedding 模型
    assert ingestion_model == query_model == "single-source-model"


def test_vector_encoder_explicit_model_still_overrides(monkeypatch):
    _set_env(monkeypatch, EMBEDDING_MODEL="env-model")
    # 显式传入 model 时仍可覆盖（仅在确需时使用）
    assert VectorEncoder(model="override").embeddings.model == "override"
