"""
用户映射工具模块。
负责用户名与用户ID之间的映射查询。
"""

import logging
from typing import Optional
from sqlalchemy import create_engine, text
import os

logger = logging.getLogger(__name__)

class UserMapper:
    """用户名映射器
    
    负责查询用户名对应的用户ID。
    """
    
    def __init__(self):
        """初始化数据库连接"""
        db_url = (
            f"mysql+pymysql://"
            f"{os.getenv('SQLBOT_DB_USER', 'root')}:"
            f"{os.getenv('SQLBOT_DB_PASSWORD', '')}@"
            f"{os.getenv('SQLBOT_DB_HOST', 'localhost')}:"
            f"{os.getenv('SQLBOT_DB_PORT', '3306')}/"
            f"{os.getenv('SQLBOT_DB_NAME', '')}"
        )
        self.engine = create_engine(db_url)
    
    def get_user_id(self, username: str) -> Optional[int]:
        """查询用户名对应的用户ID
        
        Args:
            username: 用户名
            
        Returns:
            Optional[int]: 用户ID,如果用户不存在则返回None
        """
        try:
            with self.engine.connect() as conn:
                query = text(
                    "SELECT user_id FROM user WHERE username = :username AND status = 1"
                )
                result = conn.execute(query, {"username": username}).fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"查询用户ID失败: {str(e)}")
            return None