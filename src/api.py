import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .schemas import (
    OCRRequest,
    OCRResponse,
    HealthResponse,
    ErrorResponse,
    CompanyType,
    ExtractedFields,
    CommonFields
)
from .model import ocr_model
from .database import db_client
from .utils import (
    decode_base64_image,
    save_temp_image,
    cleanup_temp_file,
    validate_image_extension,
    validate_image_size,
    resize_image_if_needed
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    gpu_info = ocr_model.get_gpu_info()

    return HealthResponse(
        status="healthy",
        model_loaded=ocr_model.is_loaded(),
        gpu_available=gpu_info.get("available", False),
        gpu_name=gpu_info.get("name"),
        gpu_memory_used_gb=gpu_info.get("memory_used_gb"),
        gpu_memory_total_gb=gpu_info.get("memory_total_gb")
    )


@router.post("/ocr", response_model=OCRResponse)
async def process_ocr(request: OCRRequest):
    """
    OCR 처리 엔드포인트 (JSON 요청)

    - image_base64: Base64 인코딩된 이미지
    - image_url: 이미지 URL
    - company_type: 업종 타입 (gym, study_room, reading_room, other)
    - custom_prompt: 커스텀 프롬프트 (선택)
    """
    if not ocr_model.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    if not request.image_base64 and not request.image_url:
        raise HTTPException(status_code=400, detail="image_base64 또는 image_url이 필요합니다.")

    start_time = time.time()
    temp_file = None

    try:
        # Base64 이미지 처리
        if request.image_base64:
            image_bytes, error = decode_base64_image(request.image_base64)
            if error:
                raise HTTPException(status_code=400, detail=error)

            # 이미지 리사이즈 (필요 시)
            image_bytes = resize_image_if_needed(image_bytes)

            # 임시 파일 저장
            temp_file = save_temp_image(image_bytes)
            if not temp_file:
                raise HTTPException(status_code=500, detail="이미지 처리 실패")

            result = ocr_model.extract_text(image_path=temp_file, custom_prompt=request.custom_prompt)
        else:
            # URL 이미지 처리
            result = ocr_model.extract_text(image_url=request.image_url, custom_prompt=request.custom_prompt)

        processing_time = (time.time() - start_time) * 1000  # ms

        # 응답 구성
        extracted = result.get("extracted_fields", {})
        common_data = extracted.get("common", {})
        custom_data = extracted.get("custom", {})

        response = OCRResponse(
            company_type=request.company_type,
            raw_text=result.get("raw_text", ""),
            extracted_fields=ExtractedFields(
                common=CommonFields(**common_data) if common_data else CommonFields(),
                custom=custom_data
            ),
            confidence=result.get("confidence", 0.5),
            processing_time_ms=round(processing_time, 2)
        )

        # DB 저장 (연결되어 있는 경우)
        if db_client.is_connected:
            await db_client.save_ocr_result(response.model_dump())

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"OCR 처리 실패: {str(e)}")
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)


@router.post("/ocr/upload", response_model=OCRResponse)
async def process_ocr_upload(
    file: UploadFile = File(...),
    company_type: CompanyType = Form(default=CompanyType.OTHER),
    custom_prompt: Optional[str] = Form(default=None)
):
    """
    OCR 처리 엔드포인트 (파일 업로드)
    """
    if not ocr_model.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    # 파일 검증
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    if not validate_image_extension(file.filename):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    start_time = time.time()
    temp_file = None

    try:
        # 파일 읽기
        image_bytes = await file.read()

        if not validate_image_size(len(image_bytes)):
            raise HTTPException(status_code=400, detail="파일 크기가 너무 큽니다.")

        # 이미지 리사이즈 (필요 시)
        image_bytes = resize_image_if_needed(image_bytes)

        # 임시 파일 저장
        temp_file = save_temp_image(image_bytes)
        if not temp_file:
            raise HTTPException(status_code=500, detail="이미지 처리 실패")

        result = ocr_model.extract_text(image_path=temp_file, custom_prompt=custom_prompt)
        processing_time = (time.time() - start_time) * 1000

        # 응답 구성
        extracted = result.get("extracted_fields", {})
        common_data = extracted.get("common", {})
        custom_data = extracted.get("custom", {})

        response = OCRResponse(
            company_type=company_type,
            raw_text=result.get("raw_text", ""),
            extracted_fields=ExtractedFields(
                common=CommonFields(**common_data) if common_data else CommonFields(),
                custom=custom_data
            ),
            confidence=result.get("confidence", 0.5),
            processing_time_ms=round(processing_time, 2)
        )

        # DB 저장
        if db_client.is_connected:
            await db_client.save_ocr_result(response.model_dump())

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"OCR 처리 실패: {str(e)}")
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)


@router.post("/db/update-url")
async def update_db_url(url: str):
    """ngrok DB URL 업데이트"""
    db_client.update_base_url(url)
    connected = await db_client.connect()
    return {
        "message": "DB URL 업데이트됨",
        "url": url,
        "connected": connected
    }
