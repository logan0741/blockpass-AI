"""
Qwen2-VL LoRA Fine-tuning 스크립트

사용법:
    python scripts/train.py --epochs 3 --batch_size 1

필요 VRAM: ~18GB (LoRA), ~12GB (QLoRA 4bit)
"""

import os
import json
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from PIL import Image

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "Qwen2-VL-7B-Instruct-KoDocOCR"
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass
class TrainConfig:
    """학습 설정"""
    # 모델
    model_path: str = str(MODEL_PATH)
    use_qlora: bool = True  # 4bit 양자화 사용 (VRAM 절약)

    # LoRA 설정
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 학습 설정
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048

    # 저장
    output_dir: str = str(OUTPUT_DIR)
    save_steps: int = 100
    logging_steps: int = 10

    # 기타
    seed: int = 42
    fp16: bool = False
    bf16: bool = True


class OCRDataset:
    """OCR 학습용 데이터셋"""

    def __init__(self, data_path: Path, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        logger.info(f"데이터 로드: {len(self.data)}개")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]

        # 이미지 로드
        image_path = item["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"이미지 로드 실패: {image_path} - {e}")
            # 빈 이미지 반환
            image = Image.new("RGB", (224, 224), color="white")

        # 대화 구성
        conversations = item["conversations"]
        user_content = conversations[0]["content"]
        assistant_content = conversations[1]["content"]

        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content}
                ]
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]

        # 프로세서로 토큰화
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 입력 처리
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

        # 라벨 설정 (입력과 동일, 패딩은 -100으로)
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return inputs


def load_model_and_processor(config: TrainConfig):
    """모델 및 프로세서 로드"""

    logger.info(f"모델 로드: {config.model_path}")

    # 양자화 설정 (QLoRA)
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

    # 모델 로드
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 프로세서 로드
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )

    # 패딩 토큰 설정
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def setup_lora(model, config: TrainConfig):
    """LoRA 설정"""

    # QLoRA 준비
    if config.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # LoRA 적용
    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"학습 가능 파라미터: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def train(config: TrainConfig):
    """학습 실행"""

    # 출력 디렉토리 생성
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model, processor = load_model_and_processor(config)

    # LoRA 설정
    model = setup_lora(model, config)

    # 데이터셋 로드
    train_path = DATA_DIR / "train.json"
    val_path = DATA_DIR / "val.json"

    if not train_path.exists():
        logger.error(f"학습 데이터 없음: {train_path}")
        logger.info("먼저 데이터셋을 준비하세요: python scripts/prepare_dataset.py")
        return

    train_dataset = OCRDataset(train_path, processor, config.max_seq_length)

    val_dataset = None
    if val_path.exists():
        val_dataset = OCRDataset(val_path, processor, config.max_seq_length)

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
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 학습 시작
    logger.info("=" * 50)
    logger.info("학습 시작")
    logger.info(f"- 에포크: {config.num_epochs}")
    logger.info(f"- 배치 크기: {config.batch_size}")
    logger.info(f"- 학습 데이터: {len(train_dataset)}개")
    logger.info(f"- LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
    logger.info("=" * 50)

    trainer.train()

    # 모델 저장
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    logger.info(f"모델 저장: {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2-VL LoRA Fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에포크")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    parser.add_argument("--lr", type=float, default=2e-5, help="학습률")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--no_qlora", action="store_true", help="QLoRA 비활성화")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="출력 디렉토리")

    args = parser.parse_args()

    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        use_qlora=not args.no_qlora,
        output_dir=args.output
    )

    train(config)
