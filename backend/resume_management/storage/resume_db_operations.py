"""
简历上传记录的关系型存储（SQLite）。

原实现依赖 MySQL（独立 server）。为让项目纯本地零 server 运行，改用 Python 内置
``sqlite3``，数据落单文件 ``RESUME_DB_PATH``（默认 ``data/app.db``）。对外函数签名不变。

注：``minio_path`` 字段名沿用历史命名，迁移后存放的是**本地文件系统相对路径**
（见 ``resume_storage_handler``）。
"""

import os
import sqlite3
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# SQLite 数据文件路径（单文件，无需 server）。
DB_PATH = os.getenv("RESUME_DB_PATH", "data/app.db")


def get_db_connection() -> Optional[sqlite3.Connection]:
    """
    获取 SQLite 数据库连接（行工厂设为 ``sqlite3.Row``，便于按列名取值）。

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


def init_resume_upload_table():
    """初始化简历上传表。"""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS resume_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_hash TEXT UNIQUE NOT NULL,
            resume_type TEXT NOT NULL CHECK (resume_type IN ('pdf', 'url')),
            file_name TEXT,
            url TEXT,
            minio_path TEXT,
            raw_content TEXT,
            upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
            is_outdated INTEGER DEFAULT 0,
            latest_resume_id TEXT
        )
        """
        )
        conn.commit()
        logger.info("Resume upload table initialized successfully.")
    except sqlite3.Error as e:
        logger.error("Error creating resume upload table: %s", e)
    finally:
        conn.close()


def store_resume_record(
    resume_hash: str,
    resume_type: str,
    file_name: Optional[str],
    url: Optional[str],
    minio_path: Optional[str],
    raw_content: str,
    is_outdated: bool = False,
    latest_resume_id: Optional[str] = None,
):
    """
    存储简历记录到数据库。

    Args:
        resume_hash (str): 简历的哈希值。
        resume_type (str): 简历类型（'pdf' 或 'url'）。
        file_name (Optional[str]): PDF文件名（仅对PDF类型有效）。
        url (Optional[str]): 简历URL（仅对URL类型有效）。
        minio_path (Optional[str]): 本地存储的文件相对路径（仅对PDF类型有效）。
        raw_content (str): 简历的原始内容。
        is_outdated (bool): 是否为过时的简历版本。
        latest_resume_id (Optional[str]): 最新版本简历的ID。
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        conn.execute(
            """
        INSERT INTO resume_uploads
        (resume_hash, resume_type, file_name, url, minio_path, raw_content, is_outdated, latest_resume_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                resume_hash,
                resume_type,
                file_name,
                url,
                minio_path,
                raw_content,
                int(is_outdated),
                latest_resume_id,
            ),
        )
        conn.commit()
        logger.info("Resume record stored successfully. Hash: %s", resume_hash)
    except sqlite3.Error as e:
        logger.error("Error storing resume record: %s", e)
        conn.rollback()
    finally:
        conn.close()


def get_resume_by_hash(resume_hash: str) -> Optional[Dict[str, Any]]:
    """
    根据哈希值检索简历记录。

    Args:
        resume_hash (str): 简历的哈希值。

    Returns:
        Optional[Dict[str, Any]]: 如果找到简历，返回包含简历信息的字典；否则返回 None。
    """
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.execute(
            "SELECT * FROM resume_uploads WHERE resume_hash = ?", (resume_hash,)
        )
        result = cursor.fetchone()
        return dict(result) if result else None
    except sqlite3.Error as e:
        logger.error("Error retrieving resume data: %s", e)
        return None
    finally:
        conn.close()


def get_minio_link(resume_hash: str) -> Optional[str]:
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.execute(
            "SELECT minio_path FROM resume_uploads WHERE resume_hash = ?",
            (resume_hash,),
        )
        result = cursor.fetchone()
        if result and result["minio_path"]:
            return result["minio_path"]
        return None
    except sqlite3.Error as e:
        logger.error("Error retrieving stored file path: %s", e)
        return None
    finally:
        conn.close()


def update_resume_version(old_resume_hash: str, new_resume_hash: str):
    """
    更新简历版本信息。

    Args:
        old_resume_hash (str): 旧版本简历的哈希值。
        new_resume_hash (str): 新版本简历的哈希值。
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        # 将旧版本标记为过时，并更新最新版本的ID
        conn.execute(
            """
            UPDATE resume_uploads
            SET is_outdated = 1, latest_resume_id = ?
            WHERE resume_hash = ?
            """,
            (new_resume_hash, old_resume_hash),
        )
        conn.commit()
        logger.info(
            "Resume version updated. Old hash: %s, New hash: %s",
            old_resume_hash,
            new_resume_hash,
        )
    except sqlite3.Error as e:
        logger.error("Error updating resume version: %s", e)
        conn.rollback()
    finally:
        conn.close()


def get_minio_path_by_id(resume_id: str) -> Optional[str]:
    """
    根据简历ID从数据库中获取本地存储路径。

    Args:
        resume_id (str): 简历ID

    Returns:
        Optional[str]: 本地存储相对路径，如果不存在则返回None
    """
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.execute(
            "SELECT minio_path FROM resume_uploads WHERE resume_hash = ?", (resume_id,)
        )
        result = cursor.fetchone()
        return result["minio_path"] if result else None
    except sqlite3.Error as e:
        logger.error("Error retrieving stored file path: %s", e)
        return None
    finally:
        conn.close()


# 初始化数据库表
init_resume_upload_table()
