from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from common.storage.file_service import FileService
from api.dependencies.auth import get_user_id
from pydantic import BaseModel

router = APIRouter(
    prefix="/files",
    tags=["文件管理"],
    responses={404: {"description": "Not found"}},
)

file_service = FileService()


class FileUploadResponse(BaseModel):
    file_url: str
    file_name: str
    file_size: int
    content_type: str


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=201,
    summary="上传文件",
    description="通用文件上传接口，返回文件访问链接"
)
async def upload_file(
        file: UploadFile = File(...),
        user_id: str = Depends(get_user_id)
) -> FileUploadResponse:
    """上传文件并返回访问链接"""
    try:
        file_path = await file_service.save_upload_file(file)

        return FileUploadResponse(
            file_url=file_path,
            file_name=file.filename,
            file_size=file.size,
            content_type=file.content_type
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")