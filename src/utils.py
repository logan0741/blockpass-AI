import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def validate_image_extension(filename: str) -> bool:
    """이미지 확장자 검증"""
    ext = Path(filename).suffix.lower()
    return ext in settings.ALLOWED_EXTENSIONS


def validate_image_size(file_size_bytes: int) -> bool:
    """이미지 크기 검증"""
    max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    return file_size_bytes <= max_size_bytes


def decode_base64_image(base64_string: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Base64 문자열을 이미지 바이트로 디코딩

    Returns:
        (이미지 바이트, 에러 메시지)
    """
    try:
        # data:image/jpeg;base64, 접두사 제거
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)

        # 크기 검증
        if not validate_image_size(len(image_bytes)):
            return None, f"이미지 크기가 {settings.MAX_IMAGE_SIZE_MB}MB를 초과합니다."

        return image_bytes, None

    except Exception as e:
        logger.error(f"Base64 디코딩 실패: {e}")
        return None, f"Base64 디코딩 실패: {str(e)}"


def save_temp_image(image_bytes: bytes, suffix: str = ".jpg") -> Optional[str]:
    """
    이미지 바이트를 임시 파일로 저장

    Returns:
        임시 파일 경로
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            return tmp.name
    except Exception as e:
        logger.error(f"임시 파일 저장 실패: {e}")
        return None


def cleanup_temp_file(file_path: str):
    """임시 파일 삭제"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"임시 파일 삭제 실패: {e}")


def get_image_info(image_bytes: bytes) -> dict:
    """이미지 정보 추출"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return {
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "size_bytes": len(image_bytes)
        }
    except Exception as e:
        logger.error(f"이미지 정보 추출 실패: {e}")
        return {}


def resize_image_if_needed(
    image_bytes: bytes,
    max_width: int = 2048,
    max_height: int = 2048
) -> bytes:
    """
    이미지가 너무 크면 리사이즈

    Args:
        image_bytes: 원본 이미지 바이트
        max_width: 최대 너비
        max_height: 최대 높이

    Returns:
        리사이즈된 이미지 바이트 (또는 원본)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))

        if image.width <= max_width and image.height <= max_height:
            return image_bytes

        # 비율 유지하며 리사이즈
        ratio = min(max_width / image.width, max_height / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))

        resized = image.resize(new_size, Image.Resampling.LANCZOS)

        # 바이트로 변환
        buffer = io.BytesIO()
        resized.save(buffer, format=image.format or "JPEG", quality=95)
        buffer.seek(0)

        logger.info(f"이미지 리사이즈: {image.size} -> {new_size}")
        return buffer.read()

    except Exception as e:
        logger.error(f"이미지 리사이즈 실패: {e}")
        return image_bytes
