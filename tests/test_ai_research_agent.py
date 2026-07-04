"""``backend.ai_research.ai_research_agent`` 的函数签名回归测试。

回归保护：``construct_subtopics`` 与 ``generate_report`` 曾以可变对象 ``[]`` 作为默认参数
（Python 经典陷阱——默认值在函数定义时创建并被所有调用共享，一旦被就地修改会跨调用泄漏）。
现改为 ``Optional[...] = None`` + 函数体内惰性初始化。此处断言默认值不再是可变对象。
"""

import inspect

import pytest

from backend.ai_research import ai_research_agent


@pytest.mark.parametrize(
    "func_name, param_name",
    [
        ("construct_subtopics", "subtopics"),
        ("generate_report", "existing_headers"),
    ],
)
def test_no_mutable_default_argument(func_name, param_name):
    func = getattr(ai_research_agent, func_name)
    param = inspect.signature(func).parameters[param_name]
    # 默认值应为 None（惰性初始化），而非共享的可变列表/字典
    assert param.default is None
    assert not isinstance(param.default, (list, dict, set))
