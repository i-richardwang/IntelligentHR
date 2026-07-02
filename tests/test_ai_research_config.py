"""``backend.ai_research.ai_research_config.Config`` 的单元测试。

聚焦 Web 抓取健壮性参数的环境变量解析与默认值，以及缺少必要
环境变量时的失败行为。
"""

import pytest


def _make_config(monkeypatch, **env):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from backend.ai_research.ai_research_config import Config

    return Config()


def test_scrape_params_default(monkeypatch):
    for key in ("SCRAPE_TIMEOUT", "SCRAPE_MAX_WORKERS", "MIN_CONTENT_LENGTH"):
        monkeypatch.delenv(key, raising=False)
    cfg = _make_config(monkeypatch)
    assert cfg.scrape_timeout == 8
    assert cfg.scrape_max_workers == 20
    assert cfg.min_content_length == 100


def test_scrape_params_env_override(monkeypatch):
    cfg = _make_config(
        monkeypatch,
        SCRAPE_TIMEOUT="15",
        SCRAPE_MAX_WORKERS="4",
        MIN_CONTENT_LENGTH="250",
    )
    assert cfg.scrape_timeout == 15
    assert cfg.scrape_max_workers == 4
    assert cfg.min_content_length == 250


def test_missing_openai_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from backend.ai_research.ai_research_config import Config

    with pytest.raises(EnvironmentError):
        Config()
