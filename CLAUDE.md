# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Korean contract document OCR service using GPU-accelerated Qwen2-VL. Extracts structured JSON from contract photos (gyms, study rooms, reading rooms) and optionally syncs with a backend via ngrok.

- **Model**: fasoo/Qwen2-VL-7B-Instruct-KoDocOCR (Korean document OCR)
- **Stack**: FastAPI + PyTorch + Transformers
- **Requirements**: NVIDIA GPU (16GB+ VRAM), CUDA 12.x, Python 3.12+

## Commands

```bash
# Activate virtual environment
source ocr_env/bin/activate

# Run server
python main.py

# Development mode (auto-reload)
DEBUG=true python main.py

# Download model (first time only, ~16GB)
huggingface-cli download fasoo/Qwen2-VL-7B-Instruct-KoDocOCR --local-dir ./models/Qwen2-VL-7B-Instruct-KoDocOCR

# Install dependencies (after PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Test health endpoint
curl http://localhost:8000/api/v1/health
```

## Architecture

```
main.py                    # FastAPI entry, lifespan loads model
├── config/settings.py     # pydantic-settings (.env loader)
└── src/
    ├── api.py             # Routes (/api/v1/*)
    ├── model.py           # OCRModel singleton (Qwen2-VL wrapper)
    ├── database.py        # ngrok-based external DB client
    ├── schemas.py         # Pydantic: OCRRequest, OCRResponse
    └── utils.py           # Image processing (base64, resize)
```

**Key Patterns**:
- Singleton instances: `ocr_model` and `db_client` (imported from modules)
- FastAPI lifespan context manager loads model at startup
- All endpoints are async
- Images >2048x2048 auto-resized to preserve VRAM

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Model/GPU status check |
| POST | `/api/v1/ocr` | OCR process (JSON body with base64) |
| POST | `/api/v1/ocr/upload` | OCR process (multipart file upload) |
| POST | `/api/v1/db/update-url` | Update ngrok DB URL at runtime |

## OCR Response Structure

```json
{
  "raw_text": "Full extracted text",
  "extracted_fields": {
    "common": { "contractor_name", "phone_number", "contract_date", "signature_exists", "address" },
    "custom": { "field_name": "value" }
  },
  "confidence": 0.95
}
```

## Core Flow

1. Server starts → `lifespan` calls `ocr_model.load()` → GPU model loaded
2. Image received → `OCRModel.extract_text()` → Qwen2-VL inference
3. Output parsed → `_parse_output()` extracts JSON from model response
4. If DB configured → Results sent to backend via ngrok URL

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Compute device |
| `TORCH_DTYPE` | `bfloat16` | Tensor precision (bfloat16/float16) |
| `DEBUG` | `false` | Enable auto-reload |
| `DB_NGROK_URL` | - | Backend ngrok URL for result sync |
| `NGROK_ENABLED` | `false` | Enable ngrok tunnel for this server |
| `NGROK_AUTH_TOKEN` | - | ngrok auth token |
| `MAX_IMAGE_SIZE_MB` | `10` | Max upload size |

## Training Scripts

Located in `scripts/`:
- `train.py` - LoRA fine-tuning for Qwen2-VL
- `train_aihub.py` - AI-Hub dataset specific training
- `prepare_dataset.py` - Dataset preparation utilities
- `extract.py` - Archive extraction helper
