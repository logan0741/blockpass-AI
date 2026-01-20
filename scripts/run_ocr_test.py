"""
Run OCR inference on Test/ images using base model + LoRA adapter.
Saves JSON outputs per image.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


BASE_MODEL = "/home/gunhee/blockpass-AI/models/Qwen2-VL-7B-Instruct-KoDocOCR"
ADAPTER_DIR = "/home/gunhee/blockpass-AI/init/training_output/final_model"
TEST_DIR = "/home/gunhee/blockpass-AI/Test"
OUT_DIR = "/home/gunhee/blockpass-AI/test_results/json"

# Reduce image token count for memory
MIN_PIXELS = 256 * 256
MAX_PIXELS = 512 * 512


def parse_output(output_text: str) -> Dict[str, Any]:
    """Best-effort JSON parse; fallback to raw text."""
    try:
        json_match = re.search(r"\{[\s\S]*\}", output_text)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return {
        "raw_text": output_text.strip(),
        "extracted_fields": {},
        "confidence": 0.0,
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
    os.makedirs(OUT_DIR, exist_ok=True)

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

    prompt = (
        "이 문서 이미지에서 모든 텍스트를 정확하게 추출해주세요.\n"
        "추출 후 다음 JSON 형식으로 정리해주세요. 필드는 가능한 한 자동으로 채우세요.\n"
        "가능하면 문서 내 '키:값' 패턴(예: 이름: 김건희)을 찾아 custom에 넣어주세요.\n\n"
        "{\n"
        '    "raw_text": "문서의 전체 텍스트",\n'
        '    "extracted_fields": {\n'
        '        "common": {\n'
        '            "contractor_name": "계약자명",\n'
        '            "phone_number": "연락처",\n'
        '            "contract_date": "계약일",\n'
        '            "signature_exists": true/false,\n'
        '            "address": "주소"\n'
        "        },\n"
        '        "custom": {\n'
        '            "기타 발견된 필드명": "값"\n'
        "        }\n"
        "    },\n"
        '    "confidence": 0.0~1.0\n'
        "}\n\n"
        "JSON만 출력하고 다른 설명은 하지 마세요."
    )

    for image_path in images:
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

        result = parse_output(output_text)
        result["_meta"] = {
            "image_path": str(image_path),
            "base_model": BASE_MODEL,
            "adapter_dir": ADAPTER_DIR,
        }

        out_path = Path(OUT_DIR) / f"{image_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
