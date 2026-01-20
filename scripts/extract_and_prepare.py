"""
Extract 20 sample images from fine-tuning data and create training dataset
Copies images, labels, and creates visualizations in Test/ folder structure
"""

import os
import json
import shutil
import cv2
import random
from pathlib import Path
import logging
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the cross-convolution analyzer
from cross_convolution_analyzer import CrossConvolutionAnalyzer, process_and_visualize_image


def check_disk_space(path: str, min_free_gb: int) -> bool:
    """Return True if free space is at least min_free_gb."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < min_free_gb:
        logger.error(
            "❌ Insufficient disk space: %.2fGB free (need at least %dGB)",
            free_gb,
            min_free_gb,
        )
        return False
    logger.info("✅ Disk space OK: %.2fGB free", free_gb)
    return True


def scan_folder_contents(root_dir: str) -> None:
    """Read all files and report counts by extension for /init step."""
    total_files = 0
    ext_counts = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            total_files += 1
            _, ext = os.path.splitext(filename)
            ext = ext.lower() or "<no_ext>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    top_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("📂 Scanned %s (files: %d)", root_dir, total_files)
    logger.info("🔎 Top extensions: %s", ", ".join(f"{k}:{v}" for k, v in top_exts))


def find_image_and_label_pairs(
    source_dir: str,
    label_dir: str,
    max_samples: int = 20
) -> List[Tuple[str, str]]:
    """
    Find matching image and label file pairs
    """
    pairs = []
    
    # Find all image files
    image_files = list(Path(source_dir).rglob("*.jpg")) + \
                  list(Path(source_dir).rglob("*.png")) + \
                  list(Path(source_dir).rglob("*.jpeg"))
    
    logger.info(f"📸 Found {len(image_files)} images in source directory")

    # Build label lookup by filename stem
    label_map = {}
    for ext in ("*.json", "*.txt", "*.xml"):
        for label_path in Path(label_dir).rglob(ext):
            label_map.setdefault(label_path.stem, str(label_path))

    # Filter to images with labels first
    labeled_images = [p for p in image_files if p.stem in label_map]
    if labeled_images:
        selected_images = random.sample(
            labeled_images,
            min(max_samples, len(labeled_images))
        )
    else:
        selected_images = random.sample(image_files, min(max_samples, len(image_files)))

    for img_path in selected_images:
        img_path_str = str(img_path)
        base_filename = img_path.stem

        label_path = label_map.get(base_filename, "")
        if not label_path:
            logger.warning(f"⚠️  No label found for {base_filename}")

        pairs.append((img_path_str, label_path))

    logger.info(f"✅ Found {len(pairs)} image-label pairs")
    return pairs[:max_samples]


def create_test_structure(
    pairs: List[Tuple[str, str]],
    output_base_dir: str,
    start_number: int = 1
) -> bool:
    """
    Create Test/Number X/ folder structure with images and visualizations
    """
    try:
        analyzer = CrossConvolutionAnalyzer()
        successful = 0
        failed = 0
        
        for idx, (img_path, label_path) in enumerate(pairs, start=start_number):
            try:
                # Create folder structure
                test_folder = os.path.join(output_base_dir, f"Number {idx}")
                photo_folder = os.path.join(test_folder, "Photo")
                json_folder = os.path.join(test_folder, "json")
                
                os.makedirs(photo_folder, exist_ok=True)
                os.makedirs(json_folder, exist_ok=True)
                
                # Process image
                if process_and_visualize_image(img_path, label_path, photo_folder, analyzer):
                    logger.info(f"✅ [{idx}] Processed: {os.path.basename(img_path)}")
                    successful += 1
                else:
                    logger.error(f"❌ [{idx}] Failed: {os.path.basename(img_path)}")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing pair {idx}: {e}")
                failed += 1
        
        logger.info(f"\n📊 Processing Summary:")
        logger.info(f"   ✅ Successful: {successful}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   📁 Output directory: {output_base_dir}")
        
        return failed == 0
        
    except Exception as e:
        logger.error(f"❌ Error in create_test_structure: {e}")
        return False


def copy_model_weights(source_model_dir: str, init_dir: str) -> bool:
    """
    Copy model weights to /init folder for separate training
    Only copies essential files to save space
    """
    try:
        essential_files = [
            "config.json",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "chat_template.json",
            "added_tokens.json",
            "generation_config.json"
        ]
        
        os.makedirs(init_dir, exist_ok=True)
        
        for filename in essential_files:
            src = os.path.join(source_model_dir, filename)
            dst = os.path.join(init_dir, filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                logger.info(f"✅ Copied: {filename}")
            else:
                logger.warning(f"⚠️  Not found: {filename}")
        
        logger.info(f"✅ Model weights copied to {init_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error copying model weights: {e}")
        return False


def copy_adapter_weight(weight_path: str, init_dir: str) -> bool:
    """Copy the current adapter weight file for separate training."""
    try:
        if not os.path.exists(weight_path):
            logger.warning("⚠️  Adapter weight not found: %s", weight_path)
            return False
        os.makedirs(init_dir, exist_ok=True)
        dst = os.path.join(init_dir, os.path.basename(weight_path))
        shutil.copy2(weight_path, dst)
        logger.info("✅ Adapter weight copied to %s", dst)
        return True
    except Exception as e:
        logger.error("❌ Error copying adapter weight: %s", e)
        return False


if __name__ == "__main__":
    import sys
    
    # Paths
    FINE_TUNE_SOURCE = "/home/gunhee/파인튜닝 데이터/원천/인.허가"
    FINE_TUNE_LABEL = "/home/gunhee/파인튜닝 데이터/라벨/인.허가"
    TEST_OUTPUT_DIR = "/home/gunhee/blockpass-AI/Test"
    MODEL_SOURCE = "/home/gunhee/blockpass-AI/models/Qwen2-VL-7B-Instruct-KoDocOCR"
    INIT_DIR = "/home/gunhee/blockpass-AI/init/model_weights"
    ADAPTER_WEIGHT = "/home/gunhee/blockpass-AI/outputs/finetune_custom/final/adapter_model.safetensors"
    MIN_FREE_GB = 10
    
    logger.info("🚀 Starting extraction and preparation process...")
    
    # Step 0: /init - read folder contents
    logger.info("\n[Step 0/4] /init - scanning fine-tuning folder contents...")
    scan_folder_contents("/home/gunhee/파인튜닝 데이터")

    # Check disk space before heavy work
    logger.info("\n[Pre-check] Disk space...")
    if not check_disk_space("/", MIN_FREE_GB):
        sys.exit(1)

    # Step 1: Find image-label pairs
    logger.info("\n[Step 1/4] Finding image-label pairs...")
    pairs = find_image_and_label_pairs(FINE_TUNE_SOURCE, FINE_TUNE_LABEL, max_samples=20)
    
    # Step 2: Create test structure with visualizations
    logger.info("\n[Step 2/4] Creating test structure with visualizations...")
    if os.path.isdir(TEST_OUTPUT_DIR):
        existing = [
            d for d in Path(TEST_OUTPUT_DIR).iterdir()
            if d.is_dir() and d.name.startswith("Number")
        ]
        if existing:
            max_num = max(int(d.name.split()[-1]) for d in existing)
            start_num = max_num + 1
        else:
            start_num = 1
    else:
        start_num = 1
    create_test_structure(pairs, TEST_OUTPUT_DIR, start_number=start_num)
    
    # Step 3: Copy model weights to init folder
    logger.info("\n[Step 3/4] Copying model weights to init folder...")
    try:
        copy_model_weights(MODEL_SOURCE, INIT_DIR)
    except Exception as e:
        logger.warning(f"⚠️  Could not copy weights (disk space): {e}")

    logger.info("\n[Step 4/4] Copying adapter weight...")
    copy_adapter_weight(ADAPTER_WEIGHT, INIT_DIR)

    logger.info("\n✅ Extraction and preparation complete!")
