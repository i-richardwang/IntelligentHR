import sqlite3
import os
from typing import Dict, Optional, Any
import json

DB_PATH = os.path.join("data", "datasets", "resume.db")


def init_all_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建 full_resume 表
    cursor.execute(
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
    conn.close()


init_all_tables()


def store_full_resume(resume_data: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
    INSERT OR REPLACE INTO full_resume 
    (resume_id, personal_info, education, work_experiences, project_experiences, 
    characteristics, experience_summary, skills_overview, resume_format, file_or_url) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    conn.close()


def get_full_resume(resume_id: str) -> Optional[Dict[str, Any]]:

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
    SELECT * FROM full_resume WHERE resume_id = ?
    """,
        (resume_id,),
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            "id": result[0],
            "personal_info": json.loads(result[1]),
            "education": json.loads(result[2]),
            "work_experiences": json.loads(result[3]),
            "project_experiences": json.loads(result[4]),
            "characteristics": result[5],
            "experience_summary": result[6],
            "skills_overview": result[7],
            "resume_format": result[8],
            "file_or_url": result[9],
        }
    return None
