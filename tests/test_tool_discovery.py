"""表格操作工具发现的回归测试。

langchain 1.x 下 ``callable(@tool 对象) == False``，若仍用 ``callable`` 过滤
``globals()`` 会筛不到任何工具，导致表格操作功能拿到空工具集。此处校验按
``invoke`` 接口过滤能正确发现工具。
"""

import backend.data_processing.table_operation.table_operations as tools_module


def test_table_operation_tools_are_discoverable():
    discovered = [
        obj
        for obj in vars(tools_module).values()
        if hasattr(obj, "invoke") and hasattr(obj, "name")
    ]
    assert len(discovered) >= 1


def test_langchain_tools_are_not_plain_callable():
    # 记录并锁定 1.x 行为：@tool 对象不是 callable，但具备 invoke 接口
    from langchain_core.tools import tool

    @tool
    def sample(x: str) -> str:
        """示例工具。"""
        return x

    assert not callable(sample)
    assert hasattr(sample, "invoke")
    assert hasattr(sample, "name")
