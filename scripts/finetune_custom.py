"""
커스텀 파인튜닝 데이터 기반 Qwen2-VL LoRA Fine-tuning 스크립트

사용법:
    # 1. 먼저 데이터 준비
    python scripts/finetune_custom.py --prepare-only

    # 2. 파인튜닝 실행
    python scripts/finetune_custom.py --train

    # 3. 한번에 실행 (준비 + 학습)
    python scripts/finetune_custom.py --prepare-only && python scripts/finetune_custom.py --train

필요 VRAM: ~12GB (QLoRA 4bit)
"""

import os
import json
import torch
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from PIL import Image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "Qwen2-VL-7B-Instruct-KoDocOCR"
OUTPUT_DIR = BASE_DIR / "outputs" / "finetune_custom"
PREPARED_DATA_DIR = BASE_DIR / "data" / "custom_prepared"

# 원본 데이터 경로
SOURCE_DATA_DIR = Path("/home/gunhee/파인튜닝 데이터")
LABEL_DIR = SOURCE_DATA_DIR / "라벨"
IMAGE_DIR = SOURCE_DATA_DIR / "원천"


@dataclass
class TrainConfig:
    """학습 설정"""
    model_path: str = str(MODEL_PATH)
    use_qlora: bool = True

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
    save_steps: int = 500
    logging_steps: int = 50

    # 기타
    seed: int = 42
    bf16: bool = True
    max_samples: int = 0  # 0 = 전체 사용


def prepare_dataset(max_samples: int = 0, train_ratio: float = 0.9) -> Dict[str, Any]:
    """
    파인튜닝 데이터를 학습용 포맷으로 변환

    Args:
        max_samples: 최대 샘플 수 (0 = 전체)
        train_ratio: 학습/검증 비율

    Returns:
        준비 결과 정보
    """
    logger.info("=" * 60)
    logger.info("데이터 준비 시작")
    logger.info(f"라벨 경로: {LABEL_DIR}")
    logger.info(f"이미지 경로: {IMAGE_DIR}")
    logger.info("=" * 60)

    # 출력 디렉토리 생성
    PREPARED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 라벨 파일 수집
    label_files = list(LABEL_DIR.rglob("*.json"))
    logger.info(f"발견된 라벨 파일: {len(label_files)}개")

    if max_samples > 0:
        label_files = label_files[:max_samples]
        logger.info(f"샘플 제한: {max_samples}개")

    # 데이터 변환
    dataset = []
    skipped = 0
    processed = 0

    for i, label_path in enumerate(label_files):
        if (i + 1) % 1000 == 0:
            logger.info(f"처리 중: {i + 1}/{len(label_files)}")

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            # 이미지 정보 추출
            image_info = label_data.get("images", [{}])[0]
            image_filename = image_info.get("image.file.name", "")

            if not image_filename:
                skipped += 1
                continue

            # 이미지 경로 구성 (라벨 경로 구조와 동일)
            relative_path = label_path.relative_to(LABEL_DIR)
            image_path = IMAGE_DIR / relative_path.parent / image_filename

            # 이미지 존재 확인
            if not image_path.exists():
                # 대체 경로 시도
                alt_image_path = IMAGE_DIR / relative_path.with_suffix(".jpg").name
                if alt_image_path.exists():
                    image_path = alt_image_path
                else:
                    skipped += 1
                    continue

            # 어노테이션에서 텍스트 추출
            annotations = label_data.get("annotations", [])
            if not annotations:
                skipped += 1
                continue

            # 바운딩 박스 기준으로 정렬 (위에서 아래, 왼쪽에서 오른쪽)
            sorted_annotations = sorted(
                annotations,
                key=lambda x: (x.get("annotation.bbox", [0, 0, 0, 0])[1],
                              x.get("annotation.bbox", [0, 0, 0, 0])[0])
            )

            # 전체 텍스트 조합
            full_text = " ".join([
                ann.get("annotation.text", "").strip()
                for ann in sorted_annotations
                if ann.get("annotation.text", "").strip()
            ])

            if not full_text:
                skipped += 1
                continue

            # 학습 데이터 형식으로 변환
            # OCR 작업: 이미지에서 텍스트 추출
            data_item = {
                "image_path": str(image_path),
                "conversations": [
                    {
                        "role": "user",
                        "content": "이 문서 이미지에서 모든 텍스트를 정확하게 추출해주세요."
                    },
                    {
                        "role": "assistant",
                        "content": full_text
                    }
                ],
                "metadata": {
                    "source_label": str(label_path),
                    "image_width": image_info.get("image.width", 0),
                    "image_height": image_info.get("image.height", 0),
                    "category": image_info.get("image.category", ""),
                    "annotation_count": len(annotations)
                }
            }

            dataset.append(data_item)
            processed += 1

        except Exception as e:
            logger.warning(f"파일 처리 실패: {label_path} - {e}")
            skipped += 1
            continue

    logger.info(f"처리 완료: {processed}개, 스킵: {skipped}개")

    if not dataset:
        logger.error("변환된 데이터가 없습니다!")
        return {"success": False, "error": "No data converted"}

    # 학습/검증 분할
    import random
    random.seed(42)
    random.shuffle(dataset)

    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    # 저장
    train_path = PREPARED_DATA_DIR / "train.json"
    val_path = PREPARED_DATA_DIR / "val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    result = {
        "success": True,
        "total_processed": processed,
        "total_skipped": skipped,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "prepared_at": datetime.now().isoformat()
    }

    # 결과 저장
    result_path = PREPARED_DATA_DIR / "prepare_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("데이터 준비 완료!")
    logger.info(f"학습 데이터: {len(train_data)}개 -> {train_path}")
    logger.info(f"검증 데이터: {len(val_data)}개 -> {val_path}")
    logger.info(f"결과 파일: {result_path}")
    logger.info("=" * 60)

    return result


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
            # 큰 이미지 리사이즈
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.warning(f"이미지 로드 실패: {image_path} - {e}")
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

        # 라벨 설정
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return inputs


def train(config: TrainConfig) -> Dict[str, Any]:
    """파인튜닝 실행"""

    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info("=" * 60)
    logger.info("파인튜닝 시작")
    logger.info("=" * 60)

    # 출력 디렉토리
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 경로 확인
    train_path = PREPARED_DATA_DIR / "train.json"
    val_path = PREPARED_DATA_DIR / "val.json"

    if not train_path.exists():
        logger.error(f"학습 데이터 없음: {train_path}")
        logger.info("먼저 --prepare-only 옵션으로 데이터를 준비하세요.")
        return {"success": False, "error": "Train data not found"}

    # 양자화 설정
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
    logger.info(f"모델 로드: {config.model_path}")
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

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

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

    model = get_peft_model(model, lora_config)

    # 파라미터 정보
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"학습 가능 파라미터: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 데이터셋 로드
    train_dataset = OCRDataset(train_path, processor, config.max_seq_length)
    val_dataset = OCRDataset(val_path, processor, config.max_seq_length) if val_path.exists() else None

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
        report_to="none",
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

    # 학습 정보 출력
    logger.info("=" * 60)
    logger.info(f"에포크: {config.num_epochs}")
    logger.info(f"배치 크기: {config.batch_size}")
    logger.info(f"학습 데이터: {len(train_dataset)}개")
    if val_dataset:
        logger.info(f"검증 데이터: {len(val_dataset)}개")
    logger.info(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
    logger.info(f"출력 경로: {output_dir}")
    logger.info("=" * 60)

    # 학습 시작
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()

    # 모델 저장
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    logger.info(f"모델 저장: {final_path}")

    # 결과 정보
    result = {
        "success": True,
        "model_path": str(final_path),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "duration_minutes": (end_time - start_time).total_seconds() / 60
    }

    # 결과 저장
    result_path = output_dir / "train_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("파인튜닝 완료!")
    logger.info(f"모델: {final_path}")
    logger.info(f"결과: {result_path}")
    logger.info(f"소요 시간: {result['duration_minutes']:.1f}분")
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="커스텀 데이터 파인튜닝")

    # 모드 선택
    parser.add_argument("--prepare-only", action="store_true", help="데이터 준비만 실행")
    parser.add_argument("--train", action="store_true", help="파인튜닝 실행")

    # 데이터 준비 옵션
    parser.add_argument("--max-samples", type=int, default=0, help="최대 샘플 수 (0=전체)")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="학습/검증 비율")

    # 학습 옵션
    parser.add_argument("--epochs", type=int, default=3, help="학습 에포크")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기")
    parser.add_argument("--lr", type=float, default=2e-5, help="학습률")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--no-qlora", action="store_true", help="QLoRA 비활성화")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="출력 디렉토리")

    args = parser.parse_args()

    if args.prepare_only:
        # 데이터 준비만
        result = prepare_dataset(
            max_samples=args.max_samples,
            train_ratio=args.train_ratio
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.train:
        # 학습 실행
        config = TrainConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            use_qlora=not args.no_qlora,
            output_dir=args.output,
            max_samples=args.max_samples
        )
        result = train(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()
        print("\n사용 예시:")
        print("  1. 데이터 준비:  python scripts/finetune_custom.py --prepare-only")
        print("  2. 학습 실행:    python scripts/finetune_custom.py --train")
        print("  3. 샘플 제한:    python scripts/finetune_custom.py --prepare-only --max-samples 1000")


if __name__ == "__main__":
    main()
