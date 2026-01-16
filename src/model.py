import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Optional, Dict, Any
import json
import re
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class OCRModel:
    """Qwen2-VL 기반 한국어 문서 OCR 모델"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = settings.DEVICE
        self._loaded = False

    def load(self) -> bool:
        """모델 로드"""
        if self._loaded:
            logger.info("모델이 이미 로드되어 있습니다.")
            return True

        try:
            logger.info(f"모델 로드 중: {settings.MODEL_PATH}")

            # dtype 설정
            dtype = torch.bfloat16 if settings.TORCH_DTYPE == "bfloat16" else torch.float16

            # 모델 로드
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                settings.MODEL_PATH,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )

            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                settings.MODEL_PATH,
                trust_remote_code=True
            )

            self._loaded = True
            logger.info("모델 로드 완료!")
            return True

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise

    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self._loaded

    def extract_text(
        self,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        이미지에서 텍스트 추출

        Args:
            image_path: 로컬 이미지 경로
            image_base64: Base64 인코딩된 이미지
            image_url: 이미지 URL
            custom_prompt: 커스텀 프롬프트

        Returns:
            추출된 텍스트와 필드 정보
        """
        if not self._loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. load() 메서드를 먼저 호출하세요.")

        # 이미지 소스 결정
        if image_path:
            image_source = f"file://{image_path}"
        elif image_base64:
            image_source = f"data:image/jpeg;base64,{image_base64}"
        elif image_url:
            image_source = image_url
        else:
            raise ValueError("이미지 소스가 제공되지 않았습니다.")

        # 기본 프롬프트 (한국어 계약서 OCR 최적화)
        default_prompt = """이 문서 이미지에서 모든 텍스트를 정확하게 추출해주세요.
추출 후 다음 JSON 형식으로 정리해주세요:

{
    "raw_text": "문서의 전체 텍스트",
    "extracted_fields": {
        "common": {
            "contractor_name": "계약자명",
            "phone_number": "연락처",
            "contract_date": "계약일",
            "signature_exists": true/false,
            "address": "주소"
        },
        "custom": {
            "기타 발견된 필드명": "값"
        }
    },
    "confidence": 0.0~1.0
}

JSON만 출력하고 다른 설명은 하지 마세요."""

        prompt = custom_prompt if custom_prompt else default_prompt

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_source},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 입력 처리
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=False
            )

        # 입력 토큰 제외하고 출력만 추출
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # JSON 파싱 시도
        result = self._parse_output(output_text)
        return result

    def _parse_output(self, output_text: str) -> Dict[str, Any]:
        """출력 텍스트를 JSON으로 파싱"""
        try:
            # JSON 블록 추출 시도
            json_match = re.search(r'\{[\s\S]*\}', output_text)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # JSON 파싱 실패 시 기본 구조 반환
        return {
            "raw_text": output_text,
            "extracted_fields": {
                "common": {},
                "custom": {}
            },
            "confidence": 0.5
        }

    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 반환"""
        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_used_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        }


# 싱글톤 인스턴스
ocr_model = OCRModel()
