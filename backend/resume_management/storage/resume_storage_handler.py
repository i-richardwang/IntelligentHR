import os
import hashlib
import aiohttp
from typing import BinaryIO
import pdfplumber
from minio import Minio
from minio.error import S3Error
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化MinIO客户端
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "resumes")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# 确保bucket存在
if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
    minio_client.make_bucket(MINIO_BUCKET_NAME)
    logger.info(f"Bucket '{MINIO_BUCKET_NAME}' created.")


def save_pdf_to_minio(file: BinaryIO) -> str:
    """
    将PDF文件保存到MinIO存储中。

    Args:
        file (BinaryIO): 要保存的PDF文件对象。

    Returns:
        str: MinIO中的文件路径。

    Raises:
        S3Error: 如果在与MinIO交互时发生错误。
    """
    try:
        file_hash = calculate_file_hash(file)
        file_path = f"pdf/{file_hash}.pdf"
        file.seek(0)  # 重置文件指针到开始
        minio_client.put_object(
            bucket_name=MINIO_BUCKET_NAME,
            object_name=file_path,
            data=file,
            length=file.getbuffer().nbytes,
            content_type="application/pdf",
        )
        logger.info(f"File saved to MinIO: {file_path}")
        return file_path
    except S3Error as e:
        logger.error(f"Error saving file to MinIO: {str(e)}")
        raise


def calculate_file_hash(file: BinaryIO) -> str:
    """
    计算文件的MD5哈希值。

    Args:
        file (BinaryIO): 要计算哈希的文件对象。

    Returns:
        str: 文件的MD5哈希值。
    """
    md5_hash = hashlib.md5()
    file.seek(0)  # 确保从文件开始读取
    for chunk in iter(lambda: file.read(4096), b""):
        md5_hash.update(chunk)
    return md5_hash.hexdigest()


def extract_text_from_pdf(file: BinaryIO) -> str:
    """
    从PDF文件中提取文本内容。

    Args:
        file (BinaryIO): PDF文件对象。

    Returns:
        str: 提取的文本内容。
    """
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


async def extract_text_from_url(url: str) -> str:
    """
    从给定的URL异步提取文本内容。

    Args:
        url (str): 要提取内容的URL。

    Returns:
        str: 提取的文本内容。

    Raises:
        Exception: 如果在请求过程中发生错误。
    """
    jina_url = f"https://r.jina.ai/{url}"
    async with aiohttp.ClientSession() as session:
        async with session.get(jina_url, ssl=False) as response:
            if response.status == 200:
                return await response.text()
            else:
                raise Exception(f"无法从URL提取内容: {url}")


def calculate_url_hash(content: str) -> str:
    """
    计算URL内容的MD5哈希值。

    Args:
        content (str): 要计算哈希的URL内容。

    Returns:
        str: 内容的MD5哈希值。
    """
    return hashlib.md5(content.encode()).hexdigest()
