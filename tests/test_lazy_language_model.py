"""语言模型延迟初始化的回归测试。

回归保护：多个模块曾在**导入期**就 ``language_model = init_language_model(...)``，
导致没配 LLM 环境变量时连 import 都失败（对展示性项目尤其致命——无法单独演示某个
功能）。改为 ``get_language_model()`` 延迟获取并缓存后，导入不再强制要求环境变量。
此处验证每个模块都暴露了该 getter，且满足“延迟 + 缓存（单实例）”契约。
"""

import importlib

import pytest

# 这些模块此前在模块级 / 类体级 eager 初始化语言模型，现已改为延迟 getter
LAZY_MODULES = [
    "backend.data_processing.table_operation.table_operation_core",
    "backend.resume_management.recommendation.recommendation_requirements",
    "backend.resume_management.recommendation.resume_search_strategy",
    "backend.resume_management.extractor.resume_extraction_core",
    "backend.text_processing.clustering.clustering_workflow",
    "backend.data_processing.data_cleaning.verification_models",
    "backend.document_check.document_check_core",
    "backend.resume_management.storage.resume_comparison",
]


@pytest.mark.parametrize("modname", LAZY_MODULES)
def test_module_exposes_lazy_getter(modname):
    # 能在测试环境（未必配置 LLM env）成功 import，且暴露延迟 getter
    mod = importlib.import_module(modname)
    assert hasattr(mod, "get_language_model")


@pytest.mark.parametrize("modname", LAZY_MODULES)
def test_get_language_model_lazy_and_cached(modname, monkeypatch):
    mod = importlib.import_module(modname)

    sentinel = object()
    calls = {"n": 0}

    def fake_init(**kwargs):
        calls["n"] += 1
        return sentinel

    # 重置缓存并替换真正的初始化函数，避免依赖真实 LLM 环境
    monkeypatch.setattr(mod, "_language_model", None)
    monkeypatch.setattr(mod, "init_language_model", fake_init)

    first = mod.get_language_model()
    second = mod.get_language_model()

    assert first is sentinel and second is sentinel
    # 只初始化一次：延迟构建 + 缓存，保持原来的单实例语义
    assert calls["n"] == 1
