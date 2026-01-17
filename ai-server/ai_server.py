import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
import requests
import uvicorn

load_dotenv()

app = FastAPI()

AI_API_KEY = os.getenv("AI_API_KEY")
BACK_API_KEY = os.getenv("BACK_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL", "http://172.10.5.40:8010")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "65535"))
SAMPLE_RESULT_PATH = os.getenv("SAMPLE_RESULT_PATH", "ocr_result_example.json")


def require_api_key(expected_key: str | None, received_key: str | None) -> None:
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if received_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_sample_result() -> list[dict]:
    try:
        with open(SAMPLE_RESULT_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return [{"name": "", "phone": ""}]


@app.post("/ai/ocr")
async def ai_ocr(
    document_id: int = Form(...),
    role: str = Form(...),
    profile_id: int = Form(...),
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-KEY"),
) -> dict:
    require_api_key(AI_API_KEY, x_api_key)

    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    result = load_sample_result()
    payload = {"document_id": document_id, "result": result}

    if not BACK_API_KEY:
        raise HTTPException(status_code=500, detail="BACK API key not configured")

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/ocr/callback",
            headers={"X-API-KEY": BACK_API_KEY},
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Callback failed: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Callback returned error")

    return {"status": "ok", "document_id": document_id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123)
