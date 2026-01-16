from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


class CompanyType(str, Enum):
    GYM = "gym"                    # 헬스장
    STUDY_ROOM = "study_room"      # 공부방
    READING_ROOM = "reading_room"  # 독서실
    OTHER = "other"                # 기타


class CommonFields(BaseModel):
    """공통 추출 필드"""
    contractor_name: Optional[str] = Field(None, description="계약자명")
    phone_number: Optional[str] = Field(None, description="연락처")
    contract_date: Optional[str] = Field(None, description="계약일")
    signature_exists: Optional[bool] = Field(None, description="서명 여부")
    address: Optional[str] = Field(None, description="주소")


class ExtractedFields(BaseModel):
    """추출된 필드 구조"""
    common: CommonFields = Field(default_factory=CommonFields)
    custom: Dict[str, Any] = Field(default_factory=dict, description="업종별 커스텀 필드")


class OCRRequest(BaseModel):
    """OCR 요청 스키마"""
    image_base64: Optional[str] = Field(None, description="Base64 인코딩된 이미지")
    image_url: Optional[str] = Field(None, description="이미지 URL")
    company_type: CompanyType = Field(default=CompanyType.OTHER, description="업종 타입")
    custom_prompt: Optional[str] = Field(None, description="커스텀 프롬프트 (선택)")


class OCRResponse(BaseModel):
    """OCR 응답 스키마"""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_type: CompanyType
    raw_text: str = Field(..., description="OCR 전체 텍스트")
    extracted_fields: ExtractedFields = Field(default_factory=ExtractedFields)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="신뢰도 점수")
    schema_version: str = Field(default="1.0.0", description="스키마 버전")
    created_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = Field(None, description="처리 시간 (ms)")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = "healthy"
    model_loaded: bool = False
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
