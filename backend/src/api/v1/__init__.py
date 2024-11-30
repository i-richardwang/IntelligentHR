from fastapi import APIRouter
from .text_analysis.routes import router as text_analysis_router

# 创建v1版本的路由器
v1_router = APIRouter(prefix="/v1")

# 注册文本分析路由
v1_router.include_router(text_analysis_router)