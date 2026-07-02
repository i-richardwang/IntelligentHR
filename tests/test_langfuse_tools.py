"""``utils.langfuse_tools`` 的单元测试。

覆盖两条关键路径：
- 未启用监控时优雅降级（不注入回调、不改动业务 config）；
- 启用监控时正确注入回调，并按 Langfuse 约定写入 ``langfuse_*`` metadata。

通过替换单例的 ``get_callback_handler`` 来模拟启用/未启用，无需真实的
Langfuse 服务端。
"""

import uuid

import utils.langfuse_tools as lt


def test_disabled_returns_empty_config(monkeypatch):
    monkeypatch.setattr(lt.langfuse_manager, "get_callback_handler", lambda: None)
    cfg = lt.get_langfuse_config(session_id="s1", tags=["a"], metadata={"step": "x"})
    # 未启用监控：既不注入回调，也不写入任何 langfuse_* 键
    assert cfg == {}


def test_disabled_preserves_existing_config(monkeypatch):
    monkeypatch.setattr(lt.langfuse_manager, "get_callback_handler", lambda: None)
    existing = {"configurable": {"thread_id": "t"}}
    cfg = lt.get_langfuse_config(session_id="s", existing_config=existing)
    assert cfg == {"configurable": {"thread_id": "t"}}
    # 返回的是浅拷贝，不应就地修改调用方传入的 config
    assert cfg is not existing


def test_enabled_injects_callback_and_metadata(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(lt.langfuse_manager, "get_callback_handler", lambda: sentinel)
    cfg = lt.get_langfuse_config(
        session_id="sess", tags=["t1", "t2"], user_id="u1", metadata={"step": "s"}
    )
    assert cfg["callbacks"] == [sentinel]
    md = cfg["metadata"]
    assert md["langfuse_session_id"] == "sess"
    assert md["langfuse_tags"] == ["t1", "t2"]
    assert md["langfuse_user_id"] == "u1"
    assert md["step"] == "s"


def test_enabled_appends_to_existing_callbacks(monkeypatch):
    sentinel = object()
    prior = object()
    monkeypatch.setattr(lt.langfuse_manager, "get_callback_handler", lambda: sentinel)
    cfg = lt.get_langfuse_config(
        session_id="s",
        existing_config={"callbacks": [prior], "configurable": {"thread_id": "x"}},
    )
    # 既有回调保留，新回调追加在后；langgraph 的 configurable 原样保留
    assert cfg["callbacks"] == [prior, sentinel]
    assert cfg["configurable"] == {"thread_id": "x"}


def test_enabled_empty_tags_omits_tag_key(monkeypatch):
    monkeypatch.setattr(lt.langfuse_manager, "get_callback_handler", lambda: object())
    cfg = lt.get_langfuse_config(session_id="s", tags=[])
    assert "langfuse_tags" not in cfg["metadata"]


def test_generate_session_id_is_unique_uuid():
    sid = lt.generate_session_id()
    uuid.UUID(sid)  # 非法 UUID 会抛异常
    assert lt.generate_session_id() != sid
