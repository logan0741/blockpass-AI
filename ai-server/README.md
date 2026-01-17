# AI Server (OCR Stub)

This is a minimal FastAPI server for OCR integration testing. It validates a
shared API key, enforces a 64KB image size limit, and sends results back to the
backend callback endpoint.

## Requirements
- Python 3.12

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn python-multipart requests python-dotenv
```

## Environment
Create `.env` from the example and set keys:
```bash
cp .env.example .env
```

- `AI_API_KEY`: key used to validate requests from the backend
- `BACK_API_KEY`: key used to call the backend callback
- `BACKEND_URL`: backend base URL (VPN IP + port)
- `MAX_IMAGE_BYTES`: set to `65535` for 64KB limit
- `SAMPLE_RESULT_PATH`: path to sample JSON file

## Run
```bash
uvicorn ai_server:app --host 0.0.0.0 --port 8123
```

## Test (from backend server)
```bash
curl -X POST "http://172.10.5.70:8123/ai/ocr" \
  -H "X-API-KEY: <AI_API_KEY>" \
  -F "document_id=1" \
  -F "role=customer" \
  -F "profile_id=1" \
  -F "image=@/path/to/image.png"
```

## Output Format
The server loads OCR output from `ocr_result_example.json` and returns it to the
backend callback. Replace this with real OCR logic later.
