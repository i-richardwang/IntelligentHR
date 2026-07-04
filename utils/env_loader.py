import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env():
    """
    加载环境变量并设置到 os.environ
    """
    # 获取项目根目录的路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 构建 .env 文件的路径
    dotenv_path = os.path.join(project_root, '.env')

    # 加载 .env 文件
    load_dotenv(dotenv_path)

    logger.info("环境变量已加载")