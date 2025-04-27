from typing import Dict, Any, Literal, Optional, List
from pydantic import BaseModel, Field


class OperationStep(BaseModel):
    """表示单个操作步骤。"""

    tool_name: str = Field(..., description="要调用的工具函数名称")
    tool_args: Dict[str, Any] = Field(..., description="调用工具函数的参数")
    output_df_names: List[str] = Field(..., description="操作结果输出的DataFrame名称列表")


class AssistantResponse(BaseModel):
    """表示AI助手的响应。"""

    next_step: Literal["need_more_info", "execute_operation", "out_of_scope"] = Field(
        ..., description="下一步操作: 需要更多信息、执行操作或超出范围"
    )
    operation: Optional[List[OperationStep]] = Field(
        None, description="当next_step为'execute_operation'时，包含要执行的操作步骤列表"
    )
    message: Optional[str] = Field(
        None,
        description="当next_step为'need_more_info'或'out_of_scope'时，包含给用户的消息",
    )
