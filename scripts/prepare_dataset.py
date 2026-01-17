"""
데이터셋 준비 스크립트

사용법:
1. AI-Hub에서 데이터셋 다운로드: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=88
2. data/raw/ 폴더에 압축 해제
3. python scripts/prepare_dataset.py 실행

또는 커스텀 데이터셋 사용:
- data/raw/images/ 에 이미지 파일
- data/raw/labels/ 에 JSON 라벨 파일
"""

import os
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def create_sample_dataset():
    """샘플 데이터셋 구조 생성 (테스트용)"""

    sample_dir = RAW_DIR / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 JSON 구조
    sample_label = {
        "image_path": "sample_001.jpg",
        "conversations": [
            {
                "role": "user",
                "content": "이 문서에서 모든 텍스트를 추출하고 JSON으로 정리해주세요."
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "raw_text": "계약서\n\n계약자: 홍길동\n연락처: 010-1234-5678\n계약일: 2026-01-16\n\n서명: [서명있음]",
                    "extracted_fields": {
                        "common": {
                            "contractor_name": "홍길동",
                            "phone_number": "010-1234-5678",
                            "contract_date": "2026-01-16",
                            "signature_exists": True,
                            "address": None
                        },
                        "custom": {
                            "계약_종류": "헬스장 회원권",
                            "금액": "100,000원"
                        }
                    },
                    "confidence": 0.95
                }, ensure_ascii=False, indent=2)
            }
        ]
    }

    # 샘플 라벨 저장
    with open(sample_dir / "sample_label.json", "w", encoding="utf-8") as f:
        json.dump(sample_label, f, ensure_ascii=False, indent=2)

    print(f"샘플 데이터셋 구조 생성됨: {sample_dir}")
    print("\n데이터셋 구조:")
    print(json.dumps(sample_label, ensure_ascii=False, indent=2))

    return sample_dir


def convert_aihub_to_training_format(aihub_dir: Path, output_dir: Path):
    """
    AI-Hub 공공행정문서 OCR 데이터셋을 학습 형식으로 변환

    AI-Hub 데이터셋 구조:
    - images/: 문서 이미지
    - labels/: JSON 라벨 (텍스트, 바운딩박스 등)
    """

    images_dir = aihub_dir / "images"
    labels_dir = aihub_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"AI-Hub 데이터셋을 찾을 수 없습니다.")
        print(f"예상 경로: {images_dir}, {labels_dir}")
        print("\n다음 단계를 따라주세요:")
        print("1. https://www.aihub.or.kr 에서 '공공행정문서 OCR' 데이터셋 다운로드")
        print(f"2. {aihub_dir} 에 압축 해제")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = []

    label_files = list(labels_dir.glob("*.json"))
    print(f"라벨 파일 수: {len(label_files)}")

    for label_file in tqdm(label_files, desc="데이터 변환 중"):
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            # 이미지 파일 찾기
            image_name = label_file.stem + ".jpg"
            image_path = images_dir / image_name

            if not image_path.exists():
                # 다른 확장자 시도
                for ext in [".png", ".jpeg", ".tif", ".tiff"]:
                    alt_path = images_dir / (label_file.stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break

            if not image_path.exists():
                continue

            # 텍스트 추출 (AI-Hub 형식에 따라 조정 필요)
            raw_text = extract_text_from_aihub_label(label_data)

            # 학습 데이터 형식으로 변환
            training_item = {
                "image_path": str(image_path),
                "conversations": [
                    {
                        "role": "user",
                        "content": "이 문서 이미지에서 모든 텍스트를 정확하게 추출해주세요."
                    },
                    {
                        "role": "assistant",
                        "content": raw_text
                    }
                ]
            }

            training_data.append(training_item)

        except Exception as e:
            print(f"오류 발생: {label_file} - {e}")
            continue

    # 데이터 분할 (train/val)
    random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.9)

    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    # 저장
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"\n변환 완료!")
    print(f"- 학습 데이터: {len(train_data)}개")
    print(f"- 검증 데이터: {len(val_data)}개")
    print(f"- 저장 위치: {output_dir}")

    return training_data


def extract_text_from_aihub_label(label_data: dict) -> str:
    """AI-Hub 라벨에서 텍스트 추출 (형식에 따라 수정 필요)"""

    # AI-Hub 공공행정문서 OCR 데이터셋 형식
    # 실제 형식에 맞게 수정 필요

    texts = []

    # 방식 1: annotations 필드
    if "annotations" in label_data:
        for ann in label_data["annotations"]:
            if "text" in ann:
                texts.append(ann["text"])

    # 방식 2: text 필드
    elif "text" in label_data:
        texts.append(label_data["text"])

    # 방식 3: ocr_result 필드
    elif "ocr_result" in label_data:
        for item in label_data["ocr_result"]:
            if "text" in item:
                texts.append(item["text"])

    return "\n".join(texts)


def convert_custom_dataset(images_dir: Path, labels_dir: Path, output_dir: Path):
    """
    커스텀 데이터셋 변환

    예상 구조:
    - images_dir/: 이미지 파일들 (001.jpg, 002.jpg, ...)
    - labels_dir/: 라벨 파일들 (001.json, 002.json, ...)

    라벨 JSON 형식:
    {
        "raw_text": "문서 전체 텍스트",
        "fields": {
            "계약자명": "홍길동",
            "연락처": "010-1234-5678",
            ...
        }
    }
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    training_data = []

    image_files = list(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]

    print(f"이미지 파일 수: {len(image_files)}")

    for image_file in tqdm(image_files, desc="커스텀 데이터 변환 중"):
        label_file = labels_dir / (image_file.stem + ".json")

        if not label_file.exists():
            print(f"라벨 없음: {image_file.name}")
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            # 응답 생성
            response = {
                "raw_text": label_data.get("raw_text", ""),
                "extracted_fields": {
                    "common": {
                        "contractor_name": label_data.get("fields", {}).get("계약자명"),
                        "phone_number": label_data.get("fields", {}).get("연락처"),
                        "contract_date": label_data.get("fields", {}).get("계약일"),
                        "signature_exists": label_data.get("fields", {}).get("서명여부"),
                        "address": label_data.get("fields", {}).get("주소")
                    },
                    "custom": {k: v for k, v in label_data.get("fields", {}).items()
                             if k not in ["계약자명", "연락처", "계약일", "서명여부", "주소"]}
                },
                "confidence": 1.0
            }

            training_item = {
                "image_path": str(image_file.absolute()),
                "conversations": [
                    {
                        "role": "user",
                        "content": "이 문서 이미지에서 모든 텍스트를 정확하게 추출하고 JSON으로 정리해주세요."
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(response, ensure_ascii=False, indent=2)
                    }
                ]
            }

            training_data.append(training_item)

        except Exception as e:
            print(f"오류: {image_file.name} - {e}")
            continue

    # 분할 및 저장
    random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.9)

    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"\n변환 완료!")
    print(f"- 학습: {len(train_data)}개, 검증: {len(val_data)}개")

    return training_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="데이터셋 준비")
    parser.add_argument("--mode", choices=["sample", "aihub", "custom"], default="sample",
                       help="sample: 샘플 구조 생성, aihub: AI-Hub 변환, custom: 커스텀 변환")
    parser.add_argument("--input", type=str, help="입력 디렉토리")

    args = parser.parse_args()

    if args.mode == "sample":
        create_sample_dataset()

    elif args.mode == "aihub":
        input_dir = Path(args.input) if args.input else RAW_DIR / "aihub"
        convert_aihub_to_training_format(input_dir, PROCESSED_DIR)

    elif args.mode == "custom":
        if not args.input:
            print("커스텀 모드는 --input 필요")
            print("예: python prepare_dataset.py --mode custom --input ./data/raw/my_data")
        else:
            input_dir = Path(args.input)
            convert_custom_dataset(
                input_dir / "images",
                input_dir / "labels",
                PROCESSED_DIR
            )
