from fastapi import APIRouter
from .file_manager.routes import router as file_manager_router
from .text_review.routes import router as text_review_router
from .text_classification.routes import router as text_classification_router
from .data_cleaning.routes import router as data_cleaning_router
from .table_manager.routes import router as table_manager_router
from .collection_manager.routes import router as collection_manager_router
from .basic_datalab.routes import router as basic_datalab_router
from .text_validity.routes import router as text_validity_router
from .sentiment_analysis.routes import router as sentiment_analysis_router
from .sensitive_detection.routes import router as sensitive_detection_router

# 创建v1版本的路由器
v1_router = APIRouter(prefix="/v1")

# 注册文件管理路由
v1_router.include_router(file_manager_router)
# 注册文本评估路由
v1_router.include_router(text_review_router)
# 注册文本分类路由
v1_router.include_router(text_classification_router)
# 注册数据清洗路由
v1_router.include_router(data_cleaning_router)
# 注册表格管理路由
v1_router.include_router(table_manager_router)
# 注册Collection管理路由
v1_router.include_router(collection_manager_router)
# 注册基础数据工坊路由
v1_router.include_router(basic_datalab_router)
# 注册文本有效性检测路由
v1_router.include_router(text_validity_router)
# 注册情感分析路由
v1_router.include_router(sentiment_analysis_router)
# 注册敏感信息检测路由
v1_router.include_router(sensitive_detection_router)
