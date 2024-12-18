from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional
import boto3
from botocore.client import Config
import logging
from config.config import config

logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """存储后端抽象基类"""
    
    @abstractmethod
    async def save_file(self, file_path: str, file_content: BinaryIO) -> str:
        """保存文件到存储后端
        
        Args:
            file_path: 文件路径/键
            file_content: 文件内容
            
        Returns:
            str: 文件的访问路径/URL
        """
        pass
    
    @abstractmethod
    def get_file(self, file_path: str) -> BinaryIO:
        """从存储后端获取文件
        
        Args:
            file_path: 文件路径/键
            
        Returns:
            BinaryIO: 文件内容
        """
        pass

class LocalStorageBackend(StorageBackend):
    """本地文件系统存储后端"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    async def save_file(self, file_path: str, file_content: BinaryIO) -> str:
        full_path = self.base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(file_content.read())
        
        return str(full_path)
    
    def get_file(self, file_path: str) -> BinaryIO:
        full_path = self.base_dir / file_path
        return open(full_path, 'rb')

class S3StorageBackend(StorageBackend):
    """S3兼容的存储后端"""
    
    def __init__(self):
        self.bucket = config.env_vars.S3_BUCKET
        self.client = boto3.client(
            's3',
            endpoint_url=config.env_vars.S3_ENDPOINT_URL,
            aws_access_key_id=config.env_vars.S3_ACCESS_KEY,
            aws_secret_access_key=config.env_vars.S3_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name=config.env_vars.S3_REGION
        )
        
    async def save_file(self, file_path: str, file_content: BinaryIO) -> str:
        try:
            self.client.upload_fileobj(file_content, self.bucket, file_path)
            return file_path
        except Exception as e:
            logger.error(f"S3上传失败: {str(e)}")
            raise
    
    def get_file(self, file_path: str) -> BinaryIO:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=file_path)
            return response['Body']
        except Exception as e:
            logger.error(f"S3下载失败: {str(e)}")
            raise

def get_storage_backend() -> StorageBackend:
    """获取配置的存储后端实例"""
    backend_type = config.storage.get('backend', 'local')
    
    if backend_type == 'local':
        return LocalStorageBackend(config.storage.base_dir)
    elif backend_type == 's3':
        return S3StorageBackend()
    else:
        raise ValueError(f"不支持的存储后端类型: {backend_type}")