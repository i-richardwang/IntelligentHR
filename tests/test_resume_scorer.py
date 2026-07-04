"""``resume_scorer.clamp_milvus_limit`` 单元测试。

回归保护 wave4 修复：Milvus 的 topk(limit) 存在硬上限（自建 Milvus 为 16384），
集合实体数超过上限、或集合为空（limit 必须 >=1）时，直接传入 search 都会抛错。
"""

from backend.resume_management.recommendation.resume_scorer import (
    clamp_milvus_limit,
    MILVUS_TOPK_LIMIT,
)


def test_normal_passthrough():
    assert clamp_milvus_limit(500) == 500


def test_zero_floored_to_one():
    assert clamp_milvus_limit(0) == 1


def test_negative_floored_to_one():
    assert clamp_milvus_limit(-10) == 1


def test_over_limit_capped():
    assert clamp_milvus_limit(20000) == MILVUS_TOPK_LIMIT


def test_exactly_at_limit():
    assert clamp_milvus_limit(MILVUS_TOPK_LIMIT) == MILVUS_TOPK_LIMIT
