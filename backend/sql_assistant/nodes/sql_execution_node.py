"""
SQL执行节点模块。
负责执行SQL查询并处理执行结果。
"""

import os
import logging
from typing import Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text, Engine

from backend.sql_assistant.states.assistant_state import SQLAssistantState

logger = logging.getLogger(__name__)


class SQLExecutor:
    """SQL执行器

    负责执行SQL查询并处理结果。
    提供查询重试机制，结果分页，错误处理等功能。
    """

    def __init__(self):
        """初始化SQL执行器"""
        self.engine = self._create_engine()

    def _create_engine(self) -> Engine:
        """创建数据库连接引擎

        Returns:
            Engine: SQLAlchemy引擎实例
        """
        db_url = (
            f"mysql+pymysql://"
            f"{os.getenv('SQLBOT_DB_USER', 'root')}:"
            f"{os.getenv('SQLBOT_DB_PASSWORD', '')}@"
            f"{os.getenv('SQLBOT_DB_HOST', 'localhost')}:"
            f"{os.getenv('SQLBOT_DB_PORT', '3306')}/"
            f"{os.getenv('SQLBOT_DB_NAME', '')}"
        )
        return create_engine(db_url)

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """执行SQL查询

        Args:
            sql_query: SQL查询语句

        Returns:
            Dict: 包含执行结果或错误信息的字典。
                 成功时包含results、columns、row_count等信息，
                 失败时包含error信息。
        """
        try:
            # 使用pandas读取SQL结果
            df = pd.read_sql_query(text(sql_query), self.engine)

            # 将结果转换为字典列表
            results = df.to_dict('records')

            # 获取列信息
            columns = list(df.columns)

            # 限制返回的记录数量
            max_rows = 100
            if len(results) > max_rows:
                results = results[:max_rows]
                truncated = True
            else:
                truncated = False

            return {
                'success': True,
                'results': results,
                'columns': columns,
                'row_count': len(df),
                'truncated': truncated,
                'error': None
            }

        except Exception as e:
            error_msg = f"SQL执行错误: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'results': None,
                'columns': None,
                'row_count': 0,
                'truncated': False,
                'error': error_msg
            }


def sql_execution_node(state: SQLAssistantState) -> dict:
    """SQL执行节点函数

    执行生成的SQL查询，处理执行结果，支持查询重试。
    使用事务确保数据一致性，支持结果分页。

    Args:
        state: 当前状态对象

    Returns:
        dict: 包含SQL执行结果的状态更新
    """
    # 获取重试计数
    retry_count = state.get("retry_count", 0)

    # 检查是否达到最大重试次数
    max_retries = 2  # 最多允许2次重试（初始执行 + 2次重试）
    if retry_count >= max_retries:
        return {
            "execution_result": {
                'success': False,
                'error': "达到最大重试次数限制"
            },
            "retry_count": retry_count
        }

    # 获取要执行的SQL
    # 优先使用错误分析节点修复后的SQL
    error_analysis = state.get("error_analysis_result", {})
    sql_source = "error_analysis" if error_analysis and error_analysis.get(
        "fixed_sql") else "generation"

    if sql_source == "error_analysis":
        sql_query = error_analysis["fixed_sql"]
    else:
        generated_sql = state.get("generated_sql", {})
        if not generated_sql:
            return {
                "execution_result": {
                    'success': False,
                    'error': "状态中未找到SQL查询语句"
                },
                "retry_count": retry_count
            }
        
        # 使用注入权限后的SQL
        sql_query = generated_sql.get('permission_controlled_sql')
        if not sql_query:
            return {
                "execution_result": {
                    'success': False,
                    'error': "状态中未找到权限控制后的SQL查询语句"
                },
                "retry_count": retry_count
            }

    try:
        # 创建执行器实例
        executor = SQLExecutor()

        # 执行SQL
        result = executor.execute_query(sql_query)

        if result['success']:
            logger.info(f"SQL执行成功: 返回 {result['row_count']} 条记录")
        else:
            logger.info(f"SQL执行失败: {result['error']}")

        # 在执行结果中添加额外信息
        result.update({
            'sql_source': sql_source,
            'executed_sql': sql_query,
            'retry_number': retry_count
        })

        # 更新状态
        return {
            "execution_result": result,
            "retry_count": retry_count + 1  # 增加重试计数
        }

    except Exception as e:
        error_msg = f"SQL执行节点出错: {str(e)}"
        logger.error(error_msg)
        return {
            "execution_result": {
                'success': False,
                'error': error_msg,
                'sql_source': sql_source,
                'executed_sql': sql_query,
                'retry_number': retry_count
            },
            "retry_count": retry_count + 1  # 即使失败也增加重试计数
        }
