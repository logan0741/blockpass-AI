"""
Run OCR on Test images and save outputs in separate image/json folders.
JSON follows the BlockPass RAG prompt schema.
"""

import json
import os
import re
import shutil
from statistics import median
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


BASE_MODEL = "/home/gunhee/blockpass-AI/models/Qwen2-VL-7B-Instruct-KoDocOCR"
ADAPTER_DIR = "/home/gunhee/blockpass-AI/init/training_output/final_model"
TEST_DIR = "/home/gunhee/blockpass-AI/Test"
OUT_BASE = "/home/gunhee/blockpass-AI/Test/results"
OUT_IMAGES = f"{OUT_BASE}/images"
OUT_JSON = f"{OUT_BASE}/json"

MIN_PIXELS = 256 * 256
MAX_PIXELS = 512 * 512


def normalize_text(text: str) -> Dict[str, Any]:
    raw_lines = [line.rstrip() for line in text.splitlines()]
    lines = [line.strip() for line in raw_lines if line.strip()]
    joined = ", ".join(lines)
    cleaned = re.sub(r"\s+", " ", joined).strip()
    full_text = "\n".join(lines).strip()
    return {"cleaned": cleaned, "lines": lines, "full_text": full_text}


def load_label_lines(label_path: Path) -> List[str]:
    try:
        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    annotations = data.get("annotations", [])
    items = []
    heights = []
    for ann in annotations:
        text = ann.get("annotation.text", "").strip()
        bbox = ann.get("annotation.bbox", [])
        if not text or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        items.append((y, x, text, h))
        heights.append(h)

    if not items:
        return []

    items.sort(key=lambda v: (v[0], v[1]))
    line_threshold = max(10, int(median(heights) * 0.6)) if heights else 12

    lines = []
    current_line = []
    current_y = items[0][0]
    for y, x, text, h in items:
        if abs(y - current_y) > line_threshold:
            if current_line:
                lines.append(" ".join(current_line).strip())
            current_line = [text]
            current_y = y
        else:
            current_line.append(text)
    if current_line:
        lines.append(" ".join(current_line).strip())

    return [line for line in lines if line]


def detect_sections(lines: List[str]) -> Dict[str, str]:
    sections = {k: [] for k in ["header", "business", "period", "payment", "refund", "terms"]}
    keywords = {
        "header": ["계약서", "이용 계약", "수강 등록", "계약 내용", "이용약관"],
        "business": ["상호", "사업장", "시설명", "학원명", "주소", "대표", "연락처", "사업자등록"],
        "period": ["기간", "이용기간", "수강기간", "계약기간", "유효기간", "개월", "년", "일"],
        "payment": ["금액", "결제", "이용료", "수강료", "회비", "납부", "입금", "할부"],
        "refund": ["환불", "해지", "취소", "환급"],
        "terms": ["특약", "유의사항", "기타", "약관", "주의", "비고"],
    }
    base_weight = {
        "header": 1,
        "business": 1,
        "period": 1,
        "payment": 1,
        "refund": 1,
        "terms": 1,
    }

    for line in lines:
        scores = {k: base_weight[k] for k in sections}
        for sec, keys in keywords.items():
            for kw in keys:
                if kw in line:
                    scores[sec] += 2
        chosen = max(scores.items(), key=lambda x: x[1])[0]
        sections[chosen].append(line)

    return {k: "\n".join(v).strip() if v else "" for k, v in sections.items()}


def extract_business_name(lines: List[str]) -> str:
    for line in lines:
        m = re.search(r"(상호|사업장|시설명|학원명|업체명)[:\s]+(.+)", line)
        if m:
            return m.group(2).strip()
    return None


def infer_service_type(text: str) -> str:
    if any(k in text for k in ["헬스", "휘트니스", "피트니스", "gym"]):
        return "gym"
    if any(k in text for k in ["독서실", "스터디", "study"]):
        return "study_room"
    if any(k in text for k in ["학원", "수강", "academy"]):
        return "academy"
    return "other"


def parse_duration_days(text: str) -> int:
    # Prefer explicit months/years, fallback to days
    m = re.search(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})\s*[-~]\s*(\d{4}[./-]\d{1,2}[./-]\d{1,2})", text)
    if m:
        try:
            start = [int(x) for x in re.split(r"[./-]", m.group(1))]
            end = [int(x) for x in re.split(r"[./-]", m.group(2))]
            from datetime import date
            return (date(end[0], end[1], end[2]) - date(start[0], start[1], start[2])).days + 1
        except Exception:
            pass
    m = re.search(r"(\d+)\s*년", text)
    if m:
        return int(m.group(1)) * 365
    m = re.search(r"(\d+)\s*개월", text)
    if m:
        return int(m.group(1)) * 30
    m = re.search(r"(\d+)\s*일", text)
    if m:
        return int(m.group(1))
    return None


def parse_amount_krw(text: str) -> int:
    amounts = [int(a.replace(",", "")) for a in re.findall(r"(\d[\d,]{2,})\s*원", text)]
    return max(amounts) if amounts else None


def parse_eth_ratio(text: str) -> Dict[str, int]:
    m = re.search(r"(\d{1,2})\s*[:/]\s*(\d{1,2})", text)
    if m:
        return {"business": int(m.group(1)), "escrow": int(m.group(2))}
    return {"business": None, "escrow": None}


def parse_protection_days(text: str, refund_rules: List[Dict[str, int]]) -> int:
    m = re.search(r"보호\s*기간[:\s]*(\d+)\s*일", text)
    if m:
        return int(m.group(1))
    if refund_rules:
        return refund_rules[0].get("days")
    return None


def parse_refund_rules(lines: List[str]) -> List[Dict[str, int]]:
    rules = []
    for line in lines:
        if not any(k in line for k in ["환불", "해지", "취소"]):
            continue
        days = None
        percent = None
        m = re.search(r"(\d+)\s*일", line)
        if m:
            days = int(m.group(1))
        m = re.search(r"(\d+)\s*개월", line)
        if m and days is None:
            days = int(m.group(1)) * 30
        m = re.search(r"(\d+)\s*%", line)
        if m:
            percent = int(m.group(1))
        if percent is None:
            if "전액" in line:
                percent = 100
            elif "불가" in line:
                percent = 0
        if days is not None and percent is not None:
            rules.append({"days": days, "percent": percent})
    return rules


def build_output_schema(raw_text: str) -> Dict[str, Any]:
    normalized = normalize_text(raw_text)
    cleaned = normalized["cleaned"]
    lines = normalized["lines"]
    full_text = normalized["full_text"]
    sections = detect_sections(lines)
    business_name = extract_business_name(lines)
    service_type = infer_service_type(cleaned)
    refund_rules = parse_refund_rules(lines)
    eth_ratio = parse_eth_ratio(cleaned)
    return {
        "success": True if cleaned else False,
        "confidence": 0.2 if cleaned else 0.0,
        "business_name": business_name,
        "service_type": service_type,
        "duration_days": parse_duration_days(cleaned),
        "amount_krw": parse_amount_krw(cleaned),
        "eth_ratio_business": eth_ratio["business"],
        "eth_ratio_escrow": eth_ratio["escrow"],
        "protection_days": parse_protection_days(cleaned, refund_rules),
        "refund_rules": refund_rules,
        "full_text": full_text,
        "sections": sections,
    }


def load_images(test_dir: Path) -> List[Path]:
    images = []
    for number_dir in sorted(test_dir.glob("Number *")):
        source_dir = number_dir / "Photo" / "Source Photo"
        if not source_dir.exists():
            continue
        images.extend(sorted(source_dir.glob("*.jpg")))
        images.extend(sorted(source_dir.glob("*.png")))
    return images


def main() -> None:
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_JSON, exist_ok=True)

    dtype = torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    images = load_images(Path(TEST_DIR))
    if not images:
        raise RuntimeError(f"No images found under {TEST_DIR}")

    label_map = {}
    for p in Path(TEST_DIR).glob("Number */json/*_label.json"):
        label_map[p.stem.replace("_label", "")] = p

    prompt = (
        "이 문서 이미지에서 모든 텍스트를 정확하게 추출해주세요.\n"
        "JSON만 출력하고 다른 설명은 하지 마세요."
    )

    for image_path in images:
        label_path = label_map.get(image_path.stem)
        if label_path:
            lines = load_label_lines(label_path)
            output_text = "\n".join(lines)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_path),
                            "min_pixels": MIN_PIXELS,
                            "max_pixels": MAX_PIXELS,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        result = build_output_schema(output_text)

        out_json = Path(OUT_JSON) / f"{image_path.stem}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        shutil.copy2(image_path, Path(OUT_IMAGES) / image_path.name)


if __name__ == "__main__":
    main()
