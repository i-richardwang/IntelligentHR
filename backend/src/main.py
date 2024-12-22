from config.config import config
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1 import v1_router
from modules.text_review.services import TaskProcessor as ReviewTaskProcessor
from modules.text_classification.services import TaskProcessor as ClassificationTaskProcessor
from modules.data_cleaning.services import TaskProcessor as CleaningTaskProcessor
from modules.text_validity.services import TaskProcessor as ValidityTaskProcessor
from modules.sentiment_analysis.services import TaskProcessor as SentimentTaskProcessor
from modules.sensitive_detection.services import TaskProcessor as SensitiveTaskProcessor
from common.utils.env_loader import load_env
from common.database.init_db import init_database
import asyncio
import uvicorn
import logging
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache


# 设置LLM缓存
# set_llm_cache(SQLiteCache(database_path="./data/llm_cache/langchain.db"))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_env()

# 创建任务处理器实例
review_processor = None
classification_processor = None
cleaning_processor = None
validity_processor = None
sentiment_processor = None
sensitive_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用程序生命周期管理器
    处理启动和关闭事件
    """
    # 启动时的初始化代码
    try:
        # 初始化数据库
        logger.info("正在初始化数据库...")
        init_database()

        # 启动文本有效性检测任务处理器
        logger.info("正在启动文本有效性检测任务处理器...")
        global validity_processor
        validity_processor = ValidityTaskProcessor()
        asyncio.create_task(validity_processor.start_processing())

        # 启动情感分析任务处理器
        logger.info("正在启动情感分析任务处理器...")
        global sentiment_processor
        sentiment_processor = SentimentTaskProcessor()
        asyncio.create_task(sentiment_processor.start_processing())

        # 启动敏感信息检测任务处理器
        logger.info("正在启动敏感信息检测任务处理器...")
        global sensitive_processor
        sensitive_processor = SensitiveTaskProcessor()
        asyncio.create_task(sensitive_processor.start_processing())

        # 启动文本评估任务处理器
        logger.info("正在启动文本评估任务处理器...")
        global review_processor
        review_processor = ReviewTaskProcessor()
        asyncio.create_task(review_processor.start_processing())

        # 启动文本分类任务处理器
        logger.info("正在启动文本分类任务处理器...")
        global classification_processor
        classification_processor = ClassificationTaskProcessor()
        asyncio.create_task(classification_processor.start_processing())

        # 启动数据清洗任务处理器
        logger.info("正在启动数据清洗任务处理器...")
        global cleaning_processor
        cleaning_processor = CleaningTaskProcessor()
        asyncio.create_task(cleaning_processor.start_processing())

        logger.info("所有任务处理器初始化完成")
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise

    yield  # 应用运行中...

    # 关闭时的清理代码
    logger.info("正在关闭应用...")
    # 这里可以添加需要的清理代码，比如关闭数据库连接等

# 创建FastAPI应用
app = FastAPI(
    title="Text Review API",
    description="文本评估服务API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册V1版本的路由
app.include_router(v1_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload
    )