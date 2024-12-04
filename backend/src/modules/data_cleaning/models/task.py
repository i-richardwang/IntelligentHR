from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.sql import func
import enum
from common.database.base import Base
from pydantic import BaseModel, Field


class TaskStatus(enum.Enum):
    """任务状态枚举类"""
    WAITING = "waiting"  # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class CleaningTask(Base):
    """数据清洗任务数据库模型"""
    __tablename__ = "cleaning_tasks"

    task_id = Column(String(36), primary_key=True, comment="任务ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.WAITING, comment="任务状态")
    entity_type = Column(String(50), nullable=False, comment="实体类型")
    source_file_url = Column(Text, nullable=False, comment="源文件URL")
    result_file_url = Column(Text, nullable=True, comment="结果文件URL")
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now(), comment="更新时间"
    )
    error_message = Column(Text, nullable=True, comment="错误信息")
    total_records = Column(Integer, nullable=True, comment="总记录数")
    processed_records = Column(Integer, nullable=True, default=0, comment="已处理记录数")
    validation_rules = Column(Text, nullable=True, comment="验证规则")
    search_enabled = Column(SQLEnum('enabled', 'disabled'), nullable=False, default='enabled', comment="是否启用搜索")
    retrieval_enabled = Column(SQLEnum('enabled', 'disabled'), nullable=False, default='enabled',
                               comment="是否启用检索")

    def to_dict(self) -> dict:
        """将模型转换为字典

        Returns:
            dict: 包含任务信息的字典
        """
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "entity_type": self.entity_type,
            "source_file_url": self.source_file_url,
            "result_file_url": self.result_file_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "progress": f"{(self.processed_records or 0)}/{self.total_records or '?'}",
            "validation_rules": self.validation_rules,
            "search_enabled": self.search_enabled,
            "retrieval_enabled": self.retrieval_enabled
        }


class TaskCreate(BaseModel):
    """任务创建请求模型"""
    entity_type: str = Field(..., description="实体类型")
    user_id: str = Field(..., description="用户ID")
    validation_rules: str | None = Field(None, description="验证规则")
    search_enabled: bool = Field(default=True, description="是否启用搜索")
    retrieval_enabled: bool = Field(default=True, description="是否启用检索")


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    user_id: str
    status: str
    entity_type: str
    source_file_url: str
    result_file_url: str | None
    created_at: datetime
    updated_at: datetime
    error_message: str | None
    total_records: int | None
    processed_records: int | None
    progress: str
    validation_rules: str | None
    search_enabled: str
    retrieval_enabled: str

    class Config:
        from_attributes = True