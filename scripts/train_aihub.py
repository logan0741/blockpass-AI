"""
AI-Hub 공공행정문서 OCR 데이터셋 학습 스크립트

사용법:
    source /home/gunhee/blockpass-AI/ocr_env/bin/activate
    cd /home/gunhee/blockpass-AI
    python scripts/train_aihub.py

옵션:
    --epochs 3          # 학습 에포크
    --batch_size 1      # 배치 크기
    --lr 2e-5           # 학습률
    --max_samples 1000  # 최대 샘플 수 (테스트용)
"""

import os
import json
import torch
import logging
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = Path("/home/gunhee/blockpass-AI")
DATA_DIR = Path("/home/gunhee/파인튜닝 데이터")
MODEL_PATH = BASE_DIR / "models" / "Qwen2-VL-7B-Instruct-KoDocOCR"
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass
class TrainConfig:
    """학습 설정"""
    # 모델
    model_path: str = str(MODEL_PATH)
    use_qlora: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 학습
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048

    # 데이터
    max_samples: Optional[int] = None  # None이면 전체 사용
    train_ratio: float = 0.9

    # 저장
    output_dir: str = str(OUTPUT_DIR)
    save_steps: int = 200
    logging_steps: int = 10

    # 기타
    seed: int = 42
    bf16: bool = True


class AIHubOCRDataset(Dataset):
    """AI-Hub 공공행정문서 OCR 데이터셋"""

    def __init__(
        self,
        data_dir: Path,
        processor,
        max_length: int = 2048,
        max_samples: Optional[int] = None
    ):
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        # 데이터 로드
        self._load_data(data_dir, max_samples)
        logger.info(f"데이터셋 로드 완료: {len(self.samples)}개")

    def _load_data(self, data_dir: Path, max_samples: Optional[int]):
        """데이터 로드"""
        label_dir = data_dir / "라벨"
        source_dir = data_dir / "원천"

        # 모든 JSON 파일 찾기
        json_files = list(label_dir.rglob("*.json"))
        logger.info(f"라벨 파일 수: {len(json_files)}")

        if max_samples:
            json_files = json_files[:max_samples]

        for json_path in tqdm(json_files, desc="데이터 로딩"):
            try:
                # JSON 파일 크기 확인 (빈 파일 스킵)
                if json_path.stat().st_size < 10:
                    continue

                with open(json_path, "r", encoding="utf-8") as f:
                    label_data = json.load(f)

                # 이미지 경로 찾기
                if "images" not in label_data or len(label_data["images"]) == 0:
                    continue

                image_info = label_data["images"][0]
                image_filename = image_info.get("image.file.name")

                if not image_filename:
                    continue

                # 이미지 경로 구성 (라벨 경로와 동일한 구조)
                relative_path = json_path.relative_to(label_dir)
                image_path = source_dir / relative_path.parent / image_filename

                if not image_path.exists():
                    # 다른 확장자 시도
                    for ext in [".jpg", ".jpeg", ".png", ".tif"]:
                        alt_path = image_path.with_suffix(ext)
                        if alt_path.exists():
                            image_path = alt_path
                            break

                if not image_path.exists():
                    continue

                # 텍스트 추출
                annotations = label_data.get("annotations", [])
                texts = [ann.get("annotation.text", "") for ann in annotations if ann.get("annotation.text")]

                if not texts:
                    continue

                full_text = " ".join(texts)

                self.samples.append({
                    "image_path": str(image_path),
                    "text": full_text,
                    "annotations": annotations
                })

            except Exception as e:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 이미지 로드
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            # 이미지 크기 제한 (메모리 절약)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                ratio = min(max_size / image.width, max_size / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.warning(f"이미지 로드 실패: {sample['image_path']}")
            image = Image.new("RGB", (224, 224), color="white")

        # 프롬프트 구성
        user_prompt = "이 문서 이미지에서 모든 텍스트를 정확하게 추출해주세요."
        assistant_response = sample["text"]

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]

        # 토큰화
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 배치 차원 제거
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # 라벨 설정
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return inputs


def load_model(config: TrainConfig):
    """모델 로드"""
    logger.info(f"모델 로드: {config.model_path}")

    # QLoRA 설정
    if config.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        logger.info("QLoRA 4bit 양자화 사용")
    else:
        bnb_config = None

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def setup_lora(model, config: TrainConfig):
    """LoRA 설정"""
    if config.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def train(config: TrainConfig):
    """학습 실행"""
    # 시드 설정
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # 출력 디렉토리
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model, processor = load_model(config)
    model = setup_lora(model, config)

    # 데이터셋 로드
    logger.info("데이터셋 로드 중...")
    full_dataset = AIHubOCRDataset(
        DATA_DIR,
        processor,
        config.max_seq_length,
        config.max_samples
    )

    # Train/Val 분할
    total = len(full_dataset)
    train_size = int(total * config.train_ratio)

    indices = list(range(total))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices) if val_indices else None

    logger.info(f"학습 데이터: {len(train_dataset)}개")
    if val_dataset:
        logger.info(f"검증 데이터: {len(val_dataset)}개")

    # 학습 인자
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=config.bf16,
        seed=config.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.save_steps if val_dataset else None,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 학습
    logger.info("=" * 50)
    logger.info("학습 시작")
    logger.info(f"에포크: {config.num_epochs}")
    logger.info(f"배치: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"학습률: {config.learning_rate}")
    logger.info(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    logger.info("=" * 50)

    trainer.train()

    # 저장
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    logger.info(f"모델 저장: {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None, help="최대 샘플 수 (테스트용)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))

    args = parser.parse_args()

    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_samples=args.max_samples,
        output_dir=args.output
    )

    train(config)
