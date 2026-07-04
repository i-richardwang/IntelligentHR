"""应用级日志配置。

Python 官方 logging 规范：库模块只应通过 ``logging.getLogger(__name__)`` 获取 logger，
**不应**在导入期配置根 logger 或添加 handler——配置日志（级别、格式、handler）是
**应用入口**的职责。本模块提供入口处一次性调用的 :func:`setup_logging`；库代码不要调用它。
"""

import logging
import os
from typing import Optional, Union

_configured = False


def setup_logging(level: Optional[Union[str, int]] = None) -> None:
    """配置应用级日志（幂等，可安全重复调用）。

    应在应用入口（如 ``frontend/app.py``）调用一次。级别默认取环境变量 ``LOG_LEVEL``
    （未设置时为 ``INFO``）。

    Args:
        level: 日志级别，可为级别名（如 ``"DEBUG"``）或整数常量；为 ``None`` 时读取
            环境变量 ``LOG_LEVEL``。
    """
    global _configured
    if _configured:
        return
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _configured = True
