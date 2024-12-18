from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, BinaryIO
from fastapi import UploadFile, HTTPException
import uuid
import logging
from config.config import config
import mimetypes
from .storage_backend import get_storage_backend, StorageBackend
import io

logger = logging.getLogger(__name__)


class FileService:
    """文件服务类

    处理文件的上传、读取和保存操作
    """

    # 文件类型映射
    MIME_TYPES = {
        '.csv': 'text/csv',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }

    def __init__(self):
        """初始化文件服务"""
        self.storage_backend = get_storage_backend()
        logger.info(f"文件服务初始化完成: 存储后端类型={type(self.storage_backend).__name__}")

    def file_exists(self, file_path: str) -> bool:
        """检查文件是否存在

        Args:
            file_path: 文件路径或URL

        Returns:
            bool: 如果文件存在且可访问返回True，否则返回False
        """
        try:
            # 如果是完整URL，提取文件路径
            if file_path.startswith(('http://', 'https://')):
                # 移除域名和bucket部分，只保留文件路径
                parts = file_path.split('/')
                if len(parts) >= 2:
                    file_path = '/'.join(parts[parts.index('uploads'):])

            # 尝试获取文件，如果成功则说明文件存在
            file = self.storage_backend.get_file(file_path)
            file.close()
            return True
        except Exception as e:
            logger.error(f"检查文件是否存在时发生错误: {str(e)}")
            return False

    def get_file_type(self, file: UploadFile) -> str:
        """获取文件类型

        Args:
            file: 上传的文件对象

        Returns:
            str: 文件的MIME类型
        """
        # 获取文件扩展名
        extension = Path(file.filename).suffix.lower()

        # 如果content_type不是octet-stream，优先使用content_type
        if file.content_type and file.content_type != 'application/octet-stream':
            return file.content_type

        # 根据扩展名判断文件类型
        if extension in self.MIME_TYPES:
            return self.MIME_TYPES[extension]

        # 使用mimetypes猜测类型
        guessed_type = mimetypes.guess_type(file.filename)[0]
        if guessed_type:
            return guessed_type

        return 'application/octet-stream'

    def validate_file(self, file: UploadFile) -> None:
        """验证上传的文件

        Args:
            file: 上传的文件对象

        Raises:
            HTTPException: 当文件验证失败时抛出
        """
        # 检查文件大小
        max_size = getattr(config.file_upload, 'max_file_size', 100 * 1024 * 1024)  # 默认100MB
        if hasattr(file, 'size') and file.size and file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制: {file.size} > {max_size} bytes"
            )

        # 检查文件类型
        content_type = self.get_file_type(file)
        # 如果配置中没有指定allowed_types，默认允许csv文件
        allowed_types = getattr(config.file_upload, 'allowed_types', ['text/csv'])

        if content_type not in allowed_types and content_type != 'application/octet-stream':
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {content_type}"
            )

        # 对于octet-stream类型，检查文件扩展名
        if content_type == 'application/octet-stream':
            extension = Path(file.filename).suffix.lower()
            if extension not in self.MIME_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型: {extension}"
                )

    async def save_upload_file(self, file: UploadFile) -> str:
        """保存上传的文件

        Args:
            file: 上传的文件对象

        Returns:
            str: 保存后的文件路径/URL
        """
        try:
            # 验证文件
            self.validate_file(file)

            # 生成文件名
            file_id = str(uuid.uuid4())
            extension = Path(file.filename).suffix
            file_path = f"uploads/{file_id}{extension}"

            # 读取文件内容
            content = await file.read()
            content_io = io.BytesIO(content)
            content_io.seek(0)

            # 使用存储后端保存文件
            file_url = await self.storage_backend.save_file(file_path, content_io)

            logger.info(f"文件已保存: {file_url}")
            return file_url

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    def read_csv_file(self, file_path: str) -> pd.DataFrame:
        """读取CSV文件

        Args:
            file_path: CSV文件路径或URL

        Returns:
            pd.DataFrame: 读取的数据框
        """
        try:
            # 如果是完整URL，提取文件路径
            if file_path.startswith(('http://', 'https://')):
                # 移除域名和bucket部分，只保留文件路径
                parts = file_path.split('/')
                if len(parts) >= 2:
                    file_path = '/'.join(parts[parts.index('uploads'):])

            file_obj = self.storage_backend.get_file(file_path)
            df = pd.read_csv(file_obj)
            file_obj.close()
            logger.info(f"成功读取CSV文件: {file_path}, 行数={len(df)}")
            return df
        except Exception as e:
            logger.error(f"读取CSV文件失败: {file_path}, 错误={str(e)}")
            raise

    async def save_results_to_csv(self, results: List[Dict[str, Any]], task_id: str) -> str:
        """保存分析结果到CSV文件

        Args:
            results: 包含原始数据和分析结果的字典列表
            task_id: 任务ID

        Returns:
            str: 结果文件路径/URL
        """
        try:
            # 创建CSV内容
            df = pd.DataFrame(results)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            
            # 转换为字节流
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            file_content = io.BytesIO(csv_bytes)
            
            # 保存文件
            file_path = f"results/{task_id}_results.csv"
            file_url = await self.storage_backend.save_file(file_path, file_content)
            
            logger.info(f"分析结果已保存: {file_url}, 行数={len(df)}")
            return file_url
        except Exception as e:
            logger.error(f"保存结果文件失败: {str(e)}")
            raise

    def get_presigned_url(self, file_path: str, expires_in: int = 3600) -> str:
        """生成预签名URL

        Args:
            file_path: 文件路径或URL
            expires_in: URL有效期（秒），默认1小时

        Returns:
            str: 预签名URL
        """
        try:
            # 如果是完整URL，提取文件路径
            if file_path.startswith(('http://', 'https://')):
                parts = file_path.split('/')
                if len(parts) >= 2:
                    file_path = '/'.join(parts[parts.index('results'):])

            # 生成预签名URL
            url = self.storage_backend.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.storage_backend.bucket,
                    'Key': file_path
                },
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"生成预签名URL失败: {str(e)}")
            raise