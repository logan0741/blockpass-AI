# BlockPass AI - OCR Core

BlockPass의 한국어 계약서 OCR 엔진을 담당하는 AI 서비스입니다.

## 핵심 모델

- **Base**: Qwen2-VL-7B-Instruct
- **Finetuned**: fasoo/Qwen2-VL-7B-Instruct-KoDocOCR
- **목표**: 한국어 문서 OCR + 계약서 문맥 구조화

## 처리 파이프라인

1) 이미지 입력 (Base64/파일 업로드)
2) Qwen2-VL OCR 추출
3) 텍스트 정규화/라인 재구성
4) 규칙 기반 문맥 분리 및 필드 추출
5) 구조화 JSON 출력

## 모델/후처리 특징

- **경량 학습**: QLoRA(4bit) + LoRA 적용
- **토큰 제어**: `min_pixels/max_pixels` 제한으로 VRAM 절감
- **문맥 분리**: 키워드 가중치 기반 섹션 분류
- **규칙 추출**: 금액/기간/환불 규정 정규식 파싱

관련 코드:
- `src/model.py` (Qwen2-VL 추론)
- `src/postprocess.py` (구조화 후처리)
- `scripts/cross_convolution_trainer.py` (LoRA/QLoRA 학습)
- `scripts/run_ocr_structured.py` (테스트 데이터 변환)

## 응답 스키마 (현재 표준)

```json
{
  "success": true,
  "confidence": 0.2,
  "business_name": "블록패스 피트니스",
  "service_type": "gym",
  "duration_days": 360,
  "amount_krw": 1200000,
  "eth_ratio_business": 50,
  "eth_ratio_escrow": 50,
  "protection_days": 30,
  "refund_rules": [
    {"days": 7, "percent": 100},
    {"days": 30, "percent": 50}
  ],
  "full_text": "OCR로 추출된 전체 텍스트...",
  "sections": {
    "header": "계약서 제목",
    "business": "사업장 정보",
    "period": "이용 기간",
    "payment": "결제 금액",
    "refund": "환불 규정",
    "terms": "특약 사항"
  }
}
```

## API

- 헬스체크: `GET /api/v1/health`
- OCR(JSON): `POST /api/v1/ocr`
- OCR(파일): `POST /api/v1/ocr/upload`

## GPU 실행 방식

- 직접 실행: `python main.py`
- Docker 실행: `blockpass-ai:local` 이미지로 GPU 컨테이너 구동

## 성능/품질 개선 포인트

- 모델: Qwen3 계열로 업그레이드 고려
- 후처리: 섹션 분리 규칙 강화(수정 가능)
- 데이터: 라벨 데이터 확대 및 도메인 특화 샘플 추가
