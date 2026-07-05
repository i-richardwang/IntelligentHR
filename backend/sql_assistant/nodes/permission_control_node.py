"""
权限控制节点模块。
负责进行SQL权限验证和权限条件注入。
"""

import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from sqlalchemy import create_engine, text, bindparam
import os
import sqlparse
from sqlparse.sql import Token, TokenList, Identifier
from pydantic import BaseModel, Field
import re

from backend.sql_assistant.states.assistant_state import SQLAssistantState

logger = logging.getLogger(__name__)


class TablePermissionConfig(BaseModel):
    """表权限配置模型"""

    table_name: str = Field(..., description="表名")
    need_dept_control: bool = Field(..., description="是否需要部门权限控制")
    dept_path_field: Optional[str] = Field(None, description="部门路径字段名")


class TableInfo(NamedTuple):
    """表信息，包含表名和别名"""

    name: str
    alias: Optional[str]


class PermissionValidator:
    """权限验证器"""

    def __init__(self):
        """初始化数据库连接"""
        # 目标业务库改用本地 SQLite（纯本地零 server）。
        db_url = f"sqlite:///{os.getenv('SQLBOT_DB_PATH', 'data/sqlbot.db')}"
        self.engine = create_engine(db_url)

    def get_all_table_names(self) -> List[str]:
        """获取数据库中所有已配置的表名"""
        query = text(
            """
            SELECT table_name 
            FROM table_permission_config 
            WHERE status = 1
        """
        )

        with self.engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result]

    def _extract_table_info(self, statement: TokenList) -> List[TableInfo]:
        """从SQL语句中提取表信息

        使用sqlparse解析SQL，提取表名和别名信息。

        Args:
            statement: SQL语句的TokenList对象

        Returns:
            List[TableInfo]: 表信息列表
        """
        tables = []
        all_table_names = set(self.get_all_table_names())

        def _process_identifier(identifier: Identifier) -> Optional[TableInfo]:
            """处理单个标识符"""
            # 获取真实的表名
            tokens = list(identifier.flatten())

            # 查找第一个匹配已知表名的token
            table_name = None
            for token in tokens:
                if token.value.lower() in {t.lower() for t in all_table_names}:
                    table_name = token.value
                    break

            if not table_name:
                return None

            # 检查是否有别名
            alias = None
            if len(tokens) > 1:
                # 最后一个token可能是别名
                last_token = tokens[-1]
                if (
                    last_token.value.lower() != table_name.lower()
                    and last_token.value.lower()
                    not in {"as", "from", "join", "where", "on", "and", "or"}
                ):
                    alias = last_token.value

            return TableInfo(table_name, alias)

        def _process_token(token):
            """递归处理token"""
            if isinstance(token, Identifier):
                table_info = _process_identifier(token)
                if table_info:
                    tables.append(table_info)
            elif isinstance(token, TokenList):
                for sub_token in token.tokens:
                    _process_token(sub_token)

        # 处理所有token
        for token in statement.tokens:
            _process_token(token)

        return tables

    def extract_table_names(self, sql: str) -> List[TableInfo]:
        """从SQL语句中提取表信息

        Args:
            sql: SQL语句

        Returns:
            List[TableInfo]: 表信息列表
        """
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                raise ValueError("SQL解析失败")

            statement = parsed[0]
            return self._extract_table_info(statement)

        except Exception as e:
            logger.error(f"提取表信息出错: {str(e)}")
            raise ValueError(f"提取表信息失败: {str(e)}")

    def get_user_accessible_tables(self, user_id: int) -> List[str]:
        """获取用户可访问的所有表名"""
        query = text(
            """
            SELECT DISTINCT tpc.table_name 
            FROM user_role ur
            JOIN role_table_permission rtp ON ur.role_id = rtp.role_id
            JOIN table_permission_config tpc ON rtp.table_permission_id = tpc.table_permission_id
            WHERE ur.user_id = :user_id
            AND tpc.status = 1
        """
        )

        with self.engine.connect() as conn:
            result = conn.execute(query, {"user_id": user_id})
            return [row[0] for row in result]

    def get_user_dept_paths(self, user_id: int) -> List[str]:
        """获取用户的部门路径列表"""
        query = text(
            """
            SELECT dept_id 
            FROM user_department 
            WHERE user_id = :user_id
        """
        )

        with self.engine.connect() as conn:
            result = conn.execute(query, {"user_id": user_id})
            return [row[0] for row in result]

    def get_table_permission_configs(
        self, table_names: List[str]
    ) -> Dict[str, TablePermissionConfig]:
        """获取表的权限配置信息"""
        if not table_names:
            return {}

        # IN 子句需用 expanding 绑定参数，SQLAlchemy 才会把序列展开为多个占位符；
        # 否则 text() 会把整个 list/tuple 当作单个参数绑定，driver 直接报错。
        query = text(
            """
            SELECT table_name, need_dept_control, dept_path_field
            FROM table_permission_config
            WHERE table_name IN :table_names
            AND status = 1
        """
        ).bindparams(bindparam("table_names", expanding=True))

        configs = {}
        with self.engine.connect() as conn:
            result = conn.execute(query, {"table_names": list(table_names)})
            for row in result:
                configs[row[0]] = TablePermissionConfig(
                    table_name=row[0],
                    need_dept_control=bool(row[1]),
                    dept_path_field=row[2],
                )
        return configs

    def _build_auth_subquery(
            self,
            table_info: TableInfo,
            dept_path_field: str,
            dept_paths: List[str]
    ) -> str:
        """构建权限验证子查询

        Args:
            table_info: 表信息（表名和别名）
            dept_path_field: 部门路径字段
            dept_paths: 用户的部门路径列表

        Returns:
            str: 构建的子查询SQL
        """
        # dept_path 字段以 ``>`` 分隔存储部门层级（如 ``1>2>3``）。旧实现用 REGEXP
        # ``(^|>)id(>|$)`` 判定某部门是否为路径中的完整一段，但业务库已本地化为
        # SQLite，其默认未注册 REGEXP 函数，该路径必然报 "no such function: REGEXP"。
        # 改用等价的 LIKE：把字段两端补 ``>`` 归一为 ``>1>2>3>`` 后，段匹配即
        # ``LIKE '%>id>%'``——既避免 id=1 命中 11，又与旧 REGEXP 语义逐段对齐，
        # 且 MySQL/SQLite 通用。
        conditions = []
        for dept_id in dept_paths:
            # dept_id 来自 user_department 表（通常为整数）；注入的子查询最终以字符串
            # 形式并入 SQL 执行、无法参数化绑定，故转义单引号做纵深防御。
            dept = str(dept_id).replace("'", "''")
            conditions.append(
                f"('>' || {dept_path_field} || '>') LIKE '%>{dept}>%'"
            )
        where_clause = " OR ".join(conditions) if conditions else "1=0"

        # 构建子查询
        subquery = f"(SELECT * FROM {table_info.name} WHERE {where_clause})"

        # 总是添加别名，如果原SQL没有指定别名，则使用原表名作为别名
        alias = table_info.alias or table_info.name
        subquery = f"{subquery} AS {alias}"

        return subquery

    def verify_and_inject_permissions(
        self, user_id: int, sql: str
    ) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """验证权限并注入权限条件"""
        try:
            # 提取SQL中的所有表信息
            table_infos = self.extract_table_names(sql)
            logger.info(f"从SQL中提取到的表信息: {table_infos}")

            # 获取所有表名
            query_tables = [info.name for info in table_infos]

            # 获取用户可访问的表
            accessible_tables = self.get_user_accessible_tables(user_id)

            # 验证表权限
            unauthorized_tables = [
                table for table in query_tables if table not in accessible_tables
            ]
            if unauthorized_tables:
                return False, None, unauthorized_tables

            # 获取表的权限配置信息
            table_configs = self.get_table_permission_configs(query_tables)

            # 获取需要部门权限控制的表
            dept_control_tables = [
                info
                for info in table_infos
                if table_configs.get(info.name)
                and table_configs[info.name].need_dept_control
            ]

            if not dept_control_tables:
                return True, sql, None

            # 获取用户的部门路径
            dept_paths = self.get_user_dept_paths(user_id)
            if not dept_paths:
                return True, sql, None

            # 处理每个需要权限控制的表
            modified_sql = sql
            for table_info in dept_control_tables:
                field = table_configs[table_info.name].dept_path_field
                if not field:
                    continue

                # 构建带权限控制的子查询
                auth_subquery = self._build_auth_subquery(table_info, field, dept_paths)

                # 替换SQL中的原表引用
                # 根据是否有别名构建不同的替换模式（表名/别名先 re.escape，防止其中的
                # 正则元字符被误解析）
                if table_info.alias:
                    pattern = (
                        rf"{re.escape(table_info.name)}\s+(?:AS\s+)?"
                        rf"{re.escape(table_info.alias)}\b"
                    )
                else:
                    # 无别名：仅替换「独立出现」的表名——要求前后都不接标识符字符或点号，
                    # 避免把 employee.name 里限定列名的 employee、或别的表名子串误替换成子查询
                    # （旧实现用 \b 会命中 employee.name 中的 employee，破坏列引用）。
                    pattern = rf"(?<![\w.]){re.escape(table_info.name)}(?![\w.])"

                # 在替换时忽略大小写；用函数式替换避免 auth_subquery 中的 \g/\1 等被
                # 当作 re 替换模板转义
                modified_sql = re.sub(
                    pattern,
                    lambda _m: auth_subquery,
                    modified_sql,
                    flags=re.IGNORECASE,
                )

            # 记录修改后的SQL，方便调试
            logger.info(f"注入权限后的SQL: {modified_sql}")
            return True, modified_sql, None

        except Exception as e:
            logger.error(f"权限验证过程出错: {str(e)}")
            return False, None, None


def permission_control_node(state: SQLAssistantState) -> dict:
    """权限控制节点函数"""
    # 获取用户ID和生成的SQL
    user_id = state.get("user_id")
    generated_sql = state.get("generated_sql", {})

    if not generated_sql or not generated_sql.get("sql_query"):
        return {
            "execution_result": {"success": False, "error": "状态中未找到生成的SQL"}
        }

    # 检查权限控制是否启用
    if os.getenv("USER_AUTH_ENABLED", "false").lower() != "true":
        return {
            "execution_result": {
                "success": True,
                "sql_query": generated_sql["sql_query"],
            },
            "generated_sql": {
                "sql_query": generated_sql["sql_query"],
                "permission_controlled_sql": generated_sql["sql_query"]
            }
        }

    if not user_id:
        return {"execution_result": {"success": False, "error": "未找到用户ID信息"}}

    try:
        # 创建权限验证器
        validator = PermissionValidator()

        # 执行权限验证和注入
        is_valid, modified_sql, unauthorized_tables = (
            validator.verify_and_inject_permissions(
                user_id=user_id, sql=generated_sql["sql_query"]
            )
        )

        if not is_valid:
            error_msg = f"权限验证失败: 无权访问表 {', '.join(unauthorized_tables or ['未知表'])}"
            return {
                "execution_result": {
                    "success": False,
                    "error": error_msg,
                    "sql_source": "permission_control",
                }
            }

        # 验证通过，更新SQL
        return {
            "generated_sql": {
                "sql_query": generated_sql["sql_query"],
                "permission_controlled_sql": modified_sql
            },
            "execution_result": {"success": True},
        }

    except Exception as e:
        error_msg = f"权限控制过程出错: {str(e)}"
        logger.error(error_msg)
        return {
            "execution_result": {
                "success": False,
                "error": error_msg,
                "sql_source": "permission_control",
            }
        }
