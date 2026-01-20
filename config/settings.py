import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # 프로젝트 경로
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: str = str(Path(__file__).resolve().parent.parent / "models" / "Qwen2-VL-7B-Instruct-KoDocOCR")

    # API 설정
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # 모델 설정
    DEVICE: str = "cuda"
    TORCH_DTYPE: str = "bfloat16"
    MAX_NEW_TOKENS: int = 2048

    # ngrok 설정
    NGROK_AUTH_TOKEN: Optional[str] = None
    NGROK_ENABLED: bool = False
    NGROK_DOMAIN: Optional[str] = None  # 커스텀 도메인 (예: blockpass.ngrok.app)

    # DB 설정 (ngrok URL로 연결)
    DB_NGROK_URL: Optional[str] = None
    DB_API_KEY: Optional[str] = None

    # 업로드 설정
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
