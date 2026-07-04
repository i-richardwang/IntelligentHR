"""
简历结构化数据的关系型存储（SQLite）。

原实现依赖 MySQL（独立 server）。为让项目纯本地零 server 运行，改用 Python 内置
``sqlite3``，与 ``resume_db_operations`` 共用同一数据文件 ``RESUME_DB_PATH``
（默认 ``data/app.db``）。对外函数签名不变。
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# SQLite 数据文件路径（与 resume_db_operations 共用）。
DB_PATH = os.getenv("RESUME_DB_PATH", "data/app.db")


def get_db_connection() -> Optional[sqlite3.Connection]:
    """
    获取 SQLite 数据库连接（行工厂设为 ``sqlite3.Row``）。

    Returns:
        Optional[sqlite3.Connection]: 连接对象，失败时返回 None。
    """
    try:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error("连接 SQLite 数据库出错: %s", e)
        return None


def init_all_tables():
    """初始化所有必要的数据库表。"""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        # 创建 full_resume 表（JSON 字段以 TEXT 存放，读写时手动 json 序列化）
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS full_resume (
            resume_id TEXT PRIMARY KEY,
            personal_info TEXT,
            education TEXT,
            work_experiences TEXT,
            project_experiences TEXT,
            characteristics TEXT,
            experience_summary TEXT,
            skills_overview TEXT,
            resume_format TEXT,
            file_or_url TEXT
        )
        """
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error("创建数据表出错: %s", e)
    finally:
        conn.close()


def store_full_resume(resume_data: Dict[str, Any]):
    """
    存储完整的简历数据到数据库（按 resume_id 幂等 upsert）。

    Args:
        resume_data (Dict[str, Any]): 包含完整简历信息的字典。
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        conn.execute(
            """
        INSERT INTO full_resume
        (resume_id, personal_info, education, work_experiences, project_experiences,
        characteristics, experience_summary, skills_overview, resume_format, file_or_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(resume_id) DO UPDATE SET
            personal_info = excluded.personal_info,
            education = excluded.education,
            work_experiences = excluded.work_experiences,
            project_experiences = excluded.project_experiences,
            characteristics = excluded.characteristics,
            experience_summary = excluded.experience_summary,
            skills_overview = excluded.skills_overview,
            resume_format = excluded.resume_format,
            file_or_url = excluded.file_or_url
        """,
            (
                resume_data["id"],
                json.dumps(resume_data["personal_info"]),
                json.dumps(resume_data["education"]),
                json.dumps(resume_data["work_experiences"]),
                json.dumps(resume_data.get("project_experiences", [])),
                resume_data.get("characteristics", ""),
                resume_data.get("experience_summary", ""),
                resume_data.get("skills_overview", ""),
                resume_data.get("resume_format", ""),
                resume_data.get("file_or_url", ""),
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error("存储简历数据出错: %s", e)
        conn.rollback()
    finally:
        conn.close()


def get_full_resume(resume_id: str) -> Optional[Dict[str, Any]]:
    """
    根据简历ID检索完整的简历数据。

    Args:
        resume_id (str): 简历的唯一标识符。

    Returns:
        Optional[Dict[str, Any]]: 如果找到简历，返回包含完整简历信息的字典；否则返回 None。
    """
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.execute(
            "SELECT * FROM full_resume WHERE resume_id = ?", (resume_id,)
        )
        result = cursor.fetchone()

        if result:
            return {
                "id": result["resume_id"],
                "personal_info": json.loads(result["personal_info"]),
                "education": json.loads(result["education"]),
                "work_experiences": json.loads(result["work_experiences"]),
                "project_experiences": json.loads(result["project_experiences"]),
                "characteristics": result["characteristics"],
                "experience_summary": result["experience_summary"],
                "skills_overview": result["skills_overview"],
                "resume_format": result["resume_format"],
                "file_or_url": result["file_or_url"],
            }
    except sqlite3.Error as e:
        logger.error("检索简历数据出错: %s", e)
    finally:
        conn.close()

    return None


# 初始化数据库表
init_all_tables()
