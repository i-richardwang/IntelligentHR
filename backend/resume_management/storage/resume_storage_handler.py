import os
import hashlib
import shutil
import aiohttp
from typing import BinaryIO
import pdfplumber
import logging

logger = logging.getLogger(__name__)

# 简历文件本地存储根目录（原用 MinIO，改为本地文件系统，纯本地零 server）。
RESUME_STORAGE_PATH = os.getenv("RESUME_STORAGE_PATH", "data/resumes")


def save_pdf_to_minio(file: BinaryIO) -> str:
    """
    将PDF文件保存到本地存储中（函数名沿用历史命名）。

    Args:
        file (BinaryIO): 要保存的PDF文件对象。

    Returns:
        str: 相对存储根目录的文件路径（形如 ``pdf/<hash>.pdf``），入库沿用此值。

    Raises:
        OSError: 写入文件失败时抛出。
    """
    try:
        file_hash = calculate_file_hash(file)
        rel_path = f"pdf/{file_hash}.pdf"
        abs_path = os.path.join(RESUME_STORAGE_PATH, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        file.seek(0)  # 重置文件指针到开始
        with open(abs_path, "wb") as out:
            shutil.copyfileobj(file, out)
        logger.info("File saved locally: %s", abs_path)
        return rel_path
    except OSError as e:
        logger.error("Error saving file locally: %s", str(e))
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
