import httpx
from typing import Optional, Dict, Any
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseClient:
    """ngrok URL 기반 외부 DB 클라이언트"""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or settings.DB_NGROK_URL
        self.api_key = api_key or settings.DB_API_KEY
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> bool:
        """DB 서버 연결"""
        if not self.base_url:
            logger.warning("DB_NGROK_URL이 설정되지 않았습니다.")
            return False

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0
            )

            # 연결 테스트
            response = await self._client.get("/health")
            if response.status_code == 200:
                logger.info(f"DB 서버 연결 성공: {self.base_url}")
                return True
            else:
                logger.warning(f"DB 서버 헬스체크 실패: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"DB 서버 연결 실패: {e}")
            return False

    async def disconnect(self):
        """연결 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def save_ocr_result(self, document_data: Dict[str, Any]) -> Optional[str]:
        """
        OCR 결과를 DB에 저장

        Args:
            document_data: OCR 결과 데이터

        Returns:
            저장된 문서 ID 또는 None
        """
        if not self._client:
            logger.error("DB 클라이언트가 연결되지 않았습니다.")
            return None

        try:
            response = await self._client.post(
                "/api/documents",
                json=document_data
            )

            if response.status_code in (200, 201):
                result = response.json()
                logger.info(f"문서 저장 성공: {result.get('document_id')}")
                return result.get("document_id")
            else:
                logger.error(f"문서 저장 실패: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"문서 저장 중 오류: {e}")
            return None

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """문서 조회"""
        if not self._client:
            return None

        try:
            response = await self._client.get(f"/api/documents/{document_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"문서 조회 중 오류: {e}")
            return None

    def update_base_url(self, new_url: str):
        """ngrok URL 업데이트 (서버 재시작 시)"""
        self.base_url = new_url
        logger.info(f"DB URL 업데이트: {new_url}")

    @property
    def is_connected(self) -> bool:
        return self._client is not None


# 싱글톤 인스턴스
db_client = DatabaseClient()
