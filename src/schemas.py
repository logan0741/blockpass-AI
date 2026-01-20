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


class RefundRule(BaseModel):
    days: int
    percent: int


class Sections(BaseModel):
    header: str = ""
    business: str = ""
    period: str = ""
    payment: str = ""
    refund: str = ""
    terms: str = ""


class StructuredOCRResponse(BaseModel):
    success: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    business_name: Optional[str] = None
    service_type: str = "other"
    duration_days: Optional[int] = None
    amount_krw: Optional[int] = None
    eth_ratio_business: Optional[int] = None
    eth_ratio_escrow: Optional[int] = None
    protection_days: Optional[int] = None
    refund_rules: List[RefundRule] = Field(default_factory=list)
    full_text: str = ""
    sections: Sections = Field(default_factory=Sections)


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
