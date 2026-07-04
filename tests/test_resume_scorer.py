"""``resume_scorer.clamp_search_limit`` 单元测试。

回归保护：检索 limit 必须落在 [1, SEARCH_TOPK_LIMIT]。集合为空时 limit 需 >=1；
上限为防御性封顶（libSQL 暴力检索本身无硬限制）。
"""

from backend.resume_management.recommendation.resume_scorer import (
    clamp_search_limit,
    SEARCH_TOPK_LIMIT,
)


def test_normal_passthrough():
    assert clamp_search_limit(500) == 500


def test_zero_floored_to_one():
    assert clamp_search_limit(0) == 1


def test_negative_floored_to_one():
    assert clamp_search_limit(-10) == 1


def test_over_limit_capped():
    assert clamp_search_limit(20000) == SEARCH_TOPK_LIMIT


def test_exactly_at_limit():
    assert clamp_search_limit(SEARCH_TOPK_LIMIT) == SEARCH_TOPK_LIMIT
