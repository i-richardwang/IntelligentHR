"""
数据清洗模块的路由定义
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from typing import List
import logging
from modules.data_cleaning.models import (
    TaskCreate,
    TaskResponse,
    TaskQueue,
    EntityConfigResponse,
)
from modules.data_cleaning.models.task import CleaningTask, TaskStatus
from modules.data_cleaning.services.entity_config_service import EntityConfigService
from common.storage.file_service import FileService
from api.dependencies.auth import get_user_id
from common.database.dependencies import get_task_db, get_app_config_db
from pydantic import BaseModel, Field
import uuid

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/data-cleaning",
    tags=["数据清洗"],
    responses={404: {"description": "未找到"}}
)

# 初始化服务
file_service = FileService()
task_queue = TaskQueue("cleaning")

class TaskCreateRequest(BaseModel):
    """任务创建请求模型"""
    file_url: str = Field(..., description="文件URL，通过文件上传接口获取")
    entity_type: str = Field(..., description="实体类型")
    search_enabled: bool = Field(True, description="是否启用搜索功能")
    retrieval_enabled: bool = Field(True, description="是否启用检索功能")

@router.get(
    "/entity-types",
    response_model=List[EntityConfigResponse],
    summary="获取支持的实体类型列表",
    description="获取系统支持的所有实体类型及其配置信息"
)
async def list_entity_types(
    user_id: str = Depends(get_user_id),
    db=Depends(get_app_config_db)
) -> List[EntityConfigResponse]:
    """获取所有支持的实体类型"""
    try:
        config_service = EntityConfigService(db)
        configs = config_service.list_configs()
        return [EntityConfigResponse.model_validate(config.to_dict()) for config in configs]
    except Exception as e:
        logger.error(f"获取实体类型列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取实体类型列表失败")

@router.post(
    "/tasks",
    response_model=TaskResponse,
    summary="创建数据清洗任务",
    description="创建新的数据清洗任务，支持实体名称标准化、验证和清洗"
)
async def create_task(
        request: TaskCreateRequest,
        response: Response,
        user_id: str = Depends(get_user_id),
        task_db=Depends(get_task_db),
        app_config_db=Depends(get_app_config_db)
) -> TaskResponse:
    """创建新的数据清洗任务"""
    try:
        # 验证实体类型是否支持
        config_service = EntityConfigService(app_config_db)
        if not config_service.is_valid_entity_type(request.entity_type):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的实体类型: {request.entity_type}"
            )

        # 验证文件URL是否有效
        if not file_service.file_exists(request.file_url):
            raise HTTPException(
                status_code=400,
                detail="文件不存在或无法访问"
            )

        # 创建任务记录
        task = CleaningTask(
            task_id=str(uuid.uuid4()),
            user_id=user_id,
            entity_type=request.entity_type,
            search_enabled='enabled' if request.search_enabled else 'disabled',
            retrieval_enabled='enabled' if request.retrieval_enabled else 'disabled',
            source_file_url=request.file_url
        )

        task_db.add(task)
        task_db.commit()
        logger.info(f"任务记录创建成功: {task.task_id}")

        # 将任务添加到队列
        await task_queue.add_task(task.task_id)

        # 设置响应状态码为201 Created
        response.status_code = 201
        return TaskResponse.model_validate(task.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail="创建任务失败")

@router.post(
    "/tasks/{task_id}/cancel",
    response_model=TaskResponse,
    summary="取消任务",
    description="取消指定的数据清洗任务"
)
async def cancel_task(
    task_id: str,
    user_id: str = Depends(get_user_id),
    db = Depends(get_task_db)
) -> TaskResponse:
    """取消指定的任务

    Args:
        task_id: 要取消的任务ID
        user_id: 用户ID
        db: 数据库会话

    Returns:
        TaskResponse: 更新后的任务信息

    Raises:
        HTTPException: 当任务不存在、无权访问或无法取消时抛出
    """
    # 检查任务是否存在且属于当前用户
    task = db.query(CleaningTask).filter(
        CleaningTask.task_id == task_id,
        CleaningTask.user_id == user_id
    ).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 检查任务是否可以被取消
    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Task cannot be cancelled in current status: {task.status.value}"
        )

    # 从TaskProcessor获取实例并取消任务
    from modules.data_cleaning.services import TaskProcessor
    processor = TaskProcessor()  # 这里利用了单例模式
    success = await processor.cancel_task(task_id)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel task")

    # 返回更新后的任务信息
    db.refresh(task)
    return TaskResponse.model_validate(task.to_dict())

@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="获取任务状态",
    description="通过任务ID获取数据清洗任务的当前状态"
)
async def get_task_status(
        task_id: str,
        user_id: str = Depends(get_user_id),
        db=Depends(get_task_db)
) -> TaskResponse:
    """获取任务状态"""
    task = db.query(CleaningTask).filter(
        CleaningTask.task_id == task_id,
        CleaningTask.user_id == user_id
    ).first()

    if not task:
        logger.warning(f"任务未找到或无权访问: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskResponse.model_validate(task.to_dict())

@router.get(
    "/tasks",
    response_model=List[TaskResponse],
    summary="获取任务列表",
    description="获取当前用户的所有数据清洗任务"
)
async def list_tasks(
        user_id: str = Depends(get_user_id),
        db=Depends(get_task_db)
) -> List[TaskResponse]:
    """获取用户的所有任务"""
    tasks = db.query(CleaningTask).filter(
        CleaningTask.user_id == user_id
    ).order_by(CleaningTask.created_at.desc()).all()

    return [TaskResponse.model_validate(task.to_dict()) for task in tasks]

@router.get(
    "/tasks/{task_id}/download",
    summary="获取清洗结果文件的下载链接",
    description="获取指定任务的数据清洗结果文件的预签名下载URL"
)
async def download_result(
        task_id: str,
        user_id: str = Depends(get_user_id),
        db=Depends(get_task_db)
):
    """获取清洗结果文件的预签名下载URL"""
    task = db.query(CleaningTask).filter(
        CleaningTask.task_id == task_id,
        CleaningTask.user_id == user_id
    ).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.result_file_url:
        raise HTTPException(status_code=400, detail="Result file not available")

    try:
        # 生成预签名URL（1小时有效期）
        presigned_url = file_service.get_presigned_url(task.result_file_url)
        return {"download_url": presigned_url}
    except Exception as e:
        logger.error(f"生成下载URL失败: {str(e)}")
        raise HTTPException(status_code=500, detail="生成下载URL失败")