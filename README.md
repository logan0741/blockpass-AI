# BlockPass AI - 한국어 계약서 OCR 시스템

헬스장, 공부방, 독서실 등의 계약 문서를 사진 기반으로 디지털화하는 AI 서비스입니다.

## 모델 정보

- **모델**: [fasoo/Qwen2-VL-7B-Instruct-KoDocOCR](https://huggingface.co/fasoo/Qwen2-VL-7B-Instruct-KoDocOCR)
- **베이스**: Qwen2-VL-7B-Instruct
- **학습 데이터**: AI-Hub 공공행정문서 OCR 데이터셋
- **특화**: 한국어 문서 OCR

## 시스템 요구사항

| 항목 | 최소 사양 |
|------|----------|
| GPU | NVIDIA GPU (VRAM 16GB 이상 권장) |
| CUDA | 12.x |
| Python | 3.12+ |
| 디스크 | 20GB 이상 (모델 16GB) |

## 프로젝트 구조

```
blockpass-AI/
├── config/
│   ├── __init__.py
│   └── settings.py          # 환경 설정
├── models/
│   └── Qwen2-VL-7B-Instruct-KoDocOCR/  # OCR 모델 (별도 다운로드)
├── src/
│   ├── __init__.py
│   ├── api.py               # FastAPI 라우터
│   ├── database.py          # ngrok DB 클라이언트
│   ├── model.py             # OCR 모델 래퍼
│   ├── schemas.py           # Pydantic 스키마
│   └── utils.py             # 이미지 처리 유틸
├── ocr_env/                 # Python 가상환경 (별도 생성)
├── .env                     # 환경변수 (별도 생성)
├── .env.example             # 환경변수 예시
├── .gitignore
├── main.py                  # FastAPI 진입점
├── requirements.txt
└── README.md
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/logan0741/blockpass-AI.git
cd blockpass-AI
```

### 2. 가상환경 생성

```bash
python3 -m venv ocr_env
source ocr_env/bin/activate
```

### 3. pip 설치 (가상환경에 pip가 없는 경우)

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
```

### 4. 의존성 설치

```bash
# PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 기타 의존성
pip install -r requirements.txt
```

### 5. 모델 다운로드

```bash
# Hugging Face CLI 사용
huggingface-cli download fasoo/Qwen2-VL-7B-Instruct-KoDocOCR \
  --local-dir ./models/Qwen2-VL-7B-Instruct-KoDocOCR
```

또는 Python으로:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fasoo/Qwen2-VL-7B-Instruct-KoDocOCR",
    local_dir="./models/Qwen2-VL-7B-Instruct-KoDocOCR"
)
```

### 6. 환경변수 설정

```bash
cp .env.example .env
# .env 파일 수정
```

## 환경변수 설명

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `API_HOST` | 서버 호스트 | `0.0.0.0` |
| `API_PORT` | 서버 포트 | `8000` |
| `DEBUG` | 디버그 모드 | `false` |
| `DEVICE` | 연산 장치 | `cuda` |
| `TORCH_DTYPE` | 텐서 타입 | `bfloat16` |
| `MAX_NEW_TOKENS` | 최대 생성 토큰 | `2048` |
| `NGROK_ENABLED` | ngrok 터널 활성화 | `false` |
| `NGROK_AUTH_TOKEN` | ngrok 인증 토큰 | - |
| `DB_NGROK_URL` | 백엔드 DB ngrok URL | - |
| `DB_API_KEY` | DB API 키 | - |
| `MAX_IMAGE_SIZE_MB` | 최대 이미지 크기 | `10` |

## 서버 실행

```bash
# 가상환경 활성화
source ocr_env/bin/activate

# 서버 실행
python main.py
```

서버가 시작되면:
- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/api/v1/health

## API 엔드포인트

### 헬스체크

```bash
GET /api/v1/health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "gpu_memory_used_gb": 14.5,
  "gpu_memory_total_gb": 24.0
}
```

### OCR 처리 (JSON)

```bash
POST /api/v1/ocr
Content-Type: application/json

{
  "image_base64": "base64_encoded_image_string",
  "company_type": "gym",
  "custom_prompt": null
}
```

**company_type 옵션:**
- `gym` - 헬스장
- `study_room` - 공부방
- `reading_room` - 독서실
- `other` - 기타

### OCR 처리 (파일 업로드)

```bash
POST /api/v1/ocr/upload
Content-Type: multipart/form-data

file: (이미지 파일)
company_type: gym
```

### OCR 응답 형식

```json
{
  "document_id": "uuid",
  "company_type": "gym",
  "raw_text": "OCR로 추출된 전체 텍스트",
  "extracted_fields": {
    "common": {
      "contractor_name": "홍길동",
      "phone_number": "010-1234-5678",
      "contract_date": "2026-01-16",
      "signature_exists": true,
      "address": "서울시 강남구..."
    },
    "custom": {
      "회원권_종류": "1개월",
      "결제금액": "100,000원",
      "시작일": "2026-01-16",
      "종료일": "2026-02-15"
    }
  },
  "confidence": 0.95,
  "schema_version": "1.0.0",
  "created_at": "2026-01-16T12:00:00",
  "processing_time_ms": 1523.45
}
```

### DB URL 업데이트

백엔드 서버의 ngrok URL이 변경되었을 때:

```bash
POST /api/v1/db/update-url?url=https://new-ngrok-url.ngrok.io
```

## 사용 예시

### Python

```python
import requests
import base64

# 이미지를 Base64로 인코딩
with open("contract.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# OCR 요청
response = requests.post(
    "http://localhost:8000/api/v1/ocr",
    json={
        "image_base64": image_base64,
        "company_type": "gym"
    }
)

result = response.json()
print(result["extracted_fields"]["common"]["contractor_name"])
```

### cURL

```bash
# 파일 업로드 방식
curl -X POST "http://localhost:8000/api/v1/ocr/upload" \
  -F "file=@contract.jpg" \
  -F "company_type=gym"
```

## 아키텍처

```
[클라이언트]
    ↓ 이미지 업로드
[GPU 서버 - FastAPI]
    ↓ OCR 처리
[Qwen2-VL 모델]
    ↓ JSON 결과
[GPU 서버]
    ↓ ngrok 터널
[백엔드 서버 - DB 저장]
    ↓
[프론트엔드 - 양식별 매핑]
```

## 지원 이미지 형식

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`
- `.pdf`

## 문제 해결

### 모델 로딩 실패

```
RuntimeError: CUDA out of memory
```

**해결:**
- GPU VRAM 16GB 이상 필요
- 다른 GPU 프로세스 종료
- `TORCH_DTYPE=float16` 설정 시도

### KeyError: 'qwen2_vl'

```
KeyError: 'qwen2_vl'
```

**해결:**
```bash
pip install --upgrade transformers
```

### ngrok 연결 실패

**해결:**
1. `NGROK_AUTH_TOKEN` 설정 확인
2. ngrok 계정에서 토큰 재발급
3. 네트워크 연결 확인

## 라이선스

- 코드: MIT License
- 모델: Apache 2.0 (fasoo/Qwen2-VL-7B-Instruct-KoDocOCR)

## 참고 자료

- [Qwen2-VL 공식 문서](https://github.com/QwenLM/Qwen2-VL)
- [Hugging Face 모델 페이지](https://huggingface.co/fasoo/Qwen2-VL-7B-Instruct-KoDocOCR)
- [AI-Hub 공공행정문서 OCR 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=88)
