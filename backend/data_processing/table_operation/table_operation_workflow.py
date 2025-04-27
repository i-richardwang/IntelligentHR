import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from backend.data_processing.table_operation.table_operation_core import (
    create_dataframe_assistant,
    process_user_query,
    create_langfuse_handler,
    record_user_feedback,
)
from backend.data_processing.table_operation.table_operations import *

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataFrameWorkflow:
    """
    管理数据框操作工作流程的类。
    """

    def __init__(self):
        """
        初始化 DataFrameWorkflow 实例。
        """
        self.assistant_chain = create_dataframe_assistant()
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.available_tools = [
            tool for tool in globals().values() if callable(tool) and hasattr(tool, 'name')
        ]
        self.conversation_history: List[Dict[str, str]] = []
        self.current_state: str = "initial"
        self.session_id: Optional[str] = None
        self.original_dataframes = set()

    def load_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        将 DataFrame 加载到工作流中，并标记为原始数据集。
        """
        self.dataframes[name] = df
        self.original_dataframes.add(name)  # 添加到原始数据集集合
        logger.info(f"Loaded DataFrame '{name}' with shape {df.shape}")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询并执行相应的操作。

        Args:
            query: 用户的查询字符串。

        Returns:
            包含操作结果的字典。
        """
        logger.info(f"Processing query: {query}")
        self.conversation_history.append({"role": "user", "content": query})

        if self.session_id is None:
            self.session_id = str(uuid.uuid4())

        dataframe_info = self.get_dataframe_info()

        full_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
        )

        result = process_user_query(
            self.assistant_chain,
            full_context,
            dataframe_info,
            self.session_id,
        )
        logger.info(f"AI response: {result}")

        # 注意：result 是一个字典，而不是 AssistantResponse 对象
        next_step = result.get("next_step")

        if next_step == "need_more_info":
            self._handle_need_more_info(result)
        elif next_step == "execute_operation":
            self._handle_execute_operation(result)
        else:  # out_of_scope
            self._handle_out_of_scope(result)

        return result

    def record_feedback(self, trace_id: str, is_useful: bool):
        # 调用 table_operation_core.py 中的函数
        record_user_feedback(trace_id, is_useful)

    def _handle_need_more_info(self, result: Dict[str, Any]) -> None:
        """
        处理需要更多信息的情况。

        Args:
            result: AI 助手的响应结果。
        """
        self.current_state = "need_more_info"
        self.conversation_history.append(
            {"role": "assistant", "content": result.get("message", "")}
        )

    def _handle_execute_operation(self, result: Dict[str, Any]) -> None:
        """
        处理执行操作的情况。

        Args:
            result: AI 助手的响应结果。
        """
        self.current_state = "ready"
        operation_steps = result.get("operation", [])

        final_results = {}
        for step in operation_steps:
            tool_name = step.get("tool_name")
            tool_args = step.get("tool_args", {})
            output_df_names = step.get("output_df_names", [])

            logger.info(f"Executing step: {tool_name} with args: {tool_args}")

            if tool_name == "stack_dataframes":
                # 特殊处理 stack_dataframes
                dataframes_with_names = []
                for df_name in tool_args.get("dataframes", []):
                    if df_name in self.dataframes:
                        dataframes_with_names.append(
                            (df_name, self.dataframes[df_name])
                        )
                tool_args["dataframes"] = dataframes_with_names
            else:
                self._replace_dataframe_names_with_objects(tool_args)

            tool_function = self._get_tool_function(tool_name)

            if tool_function:
                try:
                    step_result = tool_function.invoke(tool_args)
                    self._process_step_result(step_result, output_df_names)

                    # 如果是最后一步，保存结果
                    if step == operation_steps[-1]:
                        final_results = self._format_final_results(
                            step_result, output_df_names
                        )
                except Exception as e:
                    logger.error(f"Error executing tool: {str(e)}")
                    raise ValueError(f"执行操作时发生错误: {str(e)}")
            else:
                error_msg = f"Unknown tool: {tool_name}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        result.update(final_results)

    def _handle_out_of_scope(self, result: Dict[str, Any]) -> None:
        """
        处理超出范围的情况。

        Args:
            result: AI 助手的响应结果。
        """
        self.current_state = "out_of_scope"
        self.conversation_history.append(
            {"role": "assistant", "content": result.get("message", "")}
        )

    def _replace_dataframe_names_with_objects(self, tool_args: Dict[str, Any]) -> None:
        """
        将工具参数中的 DataFrame 名称替换为实际的 DataFrame 对象。

        Args:
            tool_args: 工具参数字典。
        """
        for arg, value in tool_args.items():
            if isinstance(value, str) and value in self.dataframes:
                tool_args[arg] = self.dataframes[value]
                logger.info(
                    f"Replaced dataframe name '{value}' with actual dataframe. "
                    f"Shape: {tool_args[arg].shape}, columns: {tool_args[arg].columns.tolist()}"
                )

    def _get_tool_function(self, tool_name: str) -> Any:
        """
        根据工具名称获取对应的工具函数。

        Args:
            tool_name: 工具函数的名称。

        Returns:
            对应的工具函数，如果未找到则返回 None。
        """
        return next(
            (tool for tool in self.available_tools if tool.name == tool_name), None
        )

    def _process_step_result(
        self, step_result: Any, output_df_names: List[str]
    ) -> None:
        """
        处理单个步骤的执行结果。

        Args:
            step_result: 步骤执行的结果。
            output_df_names: 输出 DataFrame 的名称列表。
        """
        if isinstance(step_result, tuple) and len(step_result) == len(output_df_names):
            for df, name in zip(step_result, output_df_names):
                self.dataframes[name] = df
                logger.info(f"Stored result DataFrame '{name}' with shape {df.shape}")
        elif isinstance(step_result, pd.DataFrame) and len(output_df_names) == 1:
            self.dataframes[output_df_names[0]] = step_result
            logger.info(
                f"Stored result DataFrame '{output_df_names[0]}' with shape {step_result.shape}"
            )
        else:
            logger.warning(
                f"Unexpected step result type or mismatch with output names: {type(step_result)}"
            )

    def _format_final_results(
        self, final_result: Any, output_df_names: List[str]
    ) -> Dict[str, Any]:
        """
        格式化最终结果以供返回。

        Args:
            final_result: 最后一个步骤的执行结果。
            output_df_names: 输出 DataFrame 的名称列表。

        Returns:
            格式化后的结果字典。
        """
        if isinstance(final_result, tuple) and len(final_result) == 2:
            return {"result_df1": final_result[0], "result_df2": final_result[1]}
        elif isinstance(final_result, pd.DataFrame):
            return {"result_df": final_result}
        else:
            logger.warning(f"Unexpected final result type: {type(final_result)}")
            return {"result": final_result}

    def get_original_dataframe_info(self) -> Dict[str, Dict]:
        """
        获取原始上传的数据集信息。
        """
        return {
            name: self.get_dataframe_info()[name] for name in self.original_dataframes
        }

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """
        获取指定名称的 DataFrame。

        Args:
            name: DataFrame 的名称。

        Returns:
            指定名称的 DataFrame。
        """
        return self.dataframes.get(name)

    def get_dataframe_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有 DataFrame 的信息，包括形状和数据类型。

        Returns:
            包含所有 DataFrame 信息的字典。
        """
        return {
            name: {"shape": df.shape, "dtypes": df.dtypes.to_dict()}
            for name, df in self.dataframes.items()
        }

    def get_last_message(self) -> str:
        """
        获取最后一条消息。

        Returns:
            最后一条消息的内容，如果没有消息则返回空字符串。
        """
        return (
            self.conversation_history[-1]["content"]
            if self.conversation_history
            else ""
        )

    def reset_conversation(self) -> None:
        """
        重置对话历史和当前状态。
        """
        self.conversation_history = []
        self.current_state = "initial"
        logger.info("Conversation history and state reset")
