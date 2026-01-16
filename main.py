import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import settings
from src.api import router
from src.model import ocr_model
from src.database import db_client

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 이벤트"""
    # 시작 시
    logger.info("=" * 50)
    logger.info("BlockPass AI OCR 서버 시작")
    logger.info("=" * 50)

    # 모델 로드
    logger.info("모델 로드 중...")
    ocr_model.load()
    logger.info("모델 로드 완료!")

    # GPU 정보 출력
    gpu_info = ocr_model.get_gpu_info()
    if gpu_info.get("available"):
        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"VRAM: {gpu_info['memory_used_gb']:.2f}GB / {gpu_info['memory_total_gb']:.2f}GB")

    # DB 연결 시도 (설정된 경우)
    if settings.DB_NGROK_URL:
        logger.info(f"DB 연결 시도: {settings.DB_NGROK_URL}")
        connected = await db_client.connect()
        if connected:
            logger.info("DB 연결 성공!")
        else:
            logger.warning("DB 연결 실패 - 나중에 /db/update-url로 연결 가능")

    # ngrok 터널 시작 (설정된 경우)
    if settings.NGROK_ENABLED and settings.NGROK_AUTH_TOKEN:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(settings.NGROK_AUTH_TOKEN)
            tunnel = ngrok.connect(settings.API_PORT)
            logger.info(f"ngrok 터널: {tunnel.public_url}")
        except Exception as e:
            logger.warning(f"ngrok 터널 시작 실패: {e}")

    logger.info(f"서버 주소: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info("=" * 50)

    yield  # 앱 실행 중

    # 종료 시
    logger.info("서버 종료 중...")
    await db_client.disconnect()
    logger.info("서버 종료 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="BlockPass AI OCR API",
    description="한국어 계약서 문서 디지털화 API (Qwen2-VL 기반)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router, prefix="/api/v1", tags=["OCR"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "BlockPass AI OCR",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
