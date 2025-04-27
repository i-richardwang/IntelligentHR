"""
简历 MySQL 存储模块

本模块负责简历数据的 MySQL 存储操作，包括表的初始化、数据的存储和检索。
"""

import os
import json
from typing import Dict, Optional, Any
import mysql.connector
from mysql.connector import Error

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "resume_db"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
}


def get_db_connection():
    """
    获取数据库连接。

    Returns:
        mysql.connector.connection.MySQLConnection: MySQL数据库连接对象。
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None


def init_all_tables():
    """
    初始化所有必要的数据库表。
    """
    conn = get_db_connection()
    if conn is None:
        return

    cursor = conn.cursor()

    try:
        # 创建 full_resume 表
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS full_resume (
            resume_id VARCHAR(255) PRIMARY KEY,
            personal_info JSON,
            education JSON,
            work_experiences JSON,
            project_experiences JSON,
            characteristics TEXT,
            experience_summary TEXT,
            skills_overview TEXT,
            resume_format VARCHAR(50),
            file_or_url TEXT
        )
        """
        )

        conn.commit()
    except Error as e:
        print(f"Error creating table: {e}")
    finally:
        cursor.close()
        conn.close()


def store_full_resume(resume_data: Dict[str, Any]):
    """
    存储完整的简历数据到数据库。

    Args:
        resume_data (Dict[str, Any]): 包含完整简历信息的字典。
    """
    conn = get_db_connection()
    if conn is None:
        return

    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        INSERT INTO full_resume 
        (resume_id, personal_info, education, work_experiences, project_experiences, 
        characteristics, experience_summary, skills_overview, resume_format, file_or_url) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        personal_info = VALUES(personal_info),
        education = VALUES(education),
        work_experiences = VALUES(work_experiences),
        project_experiences = VALUES(project_experiences),
        characteristics = VALUES(characteristics),
        experience_summary = VALUES(experience_summary),
        skills_overview = VALUES(skills_overview),
        resume_format = VALUES(resume_format),
        file_or_url = VALUES(file_or_url)
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
    except Error as e:
        print(f"Error storing resume data: {e}")
        conn.rollback()
    finally:
        cursor.close()
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

    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM full_resume WHERE resume_id = %s", (resume_id,))
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
    except Error as e:
        print(f"Error retrieving resume data: {e}")
    finally:
        cursor.close()
        conn.close()

    return None


# 初始化数据库表
init_all_tables()
