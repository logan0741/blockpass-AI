"""
Cross-Convolution Enhanced Training Script
Trains Qwen2-VL model with cross-shape convolution analysis
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossConvolutionTrainer:
    """
    Enhanced trainer with cross-convolution analysis
    """

    def __init__(self, model_path: str, init_dir: str, device: str = "cuda"):
        self.model_path = model_path
        self.init_dir = init_dir
        self.device = device
        self.model = None
        self.processor = None
        self.use_qlora = True
        self.min_pixels = 256 * 256
        self.max_pixels = 512 * 512
        
    def load_model(self):
        """Load model and processor"""
        logger.info(f"📦 Loading model from {self.model_path}...")
        
        dtype = torch.float16  # lower memory than bfloat16

        bnb_config = None
        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        logger.info("✅ Model loaded successfully!")
        # Reduce memory during training
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        if self.use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info("🔧 LoRA enabled: %d/%d trainable (%.2f%%)", trainable, total, 100 * trainable / total)
        return True
    
    def load_training_data(self, data_dir: str) -> Dataset:
        """
        Load training data from Test/Number X/ structure
        Includes cross-convolution features
        """
        data_items = []
        
        # Find all Number X folders
        test_dirs = sorted(
            [d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith("Number")],
            key=lambda x: int(x.name.split()[-1])
        )
        
        logger.info(f"📂 Found {len(test_dirs)} training folders")
        
        for test_dir in test_dirs:
            try:
                photo_dir = test_dir / "Photo"
                json_dir = test_dir / "json"
                
                # Find images
                images = list(photo_dir.glob("**/Source Photo/*.jpg")) + \
                        list(photo_dir.glob("**/Source Photo/*.png"))
                
                for img_path in images:
                    base_name = img_path.stem
                    
                    # Load analysis JSON
                    analysis_json = json_dir / f"{base_name}_analysis.json"
                    analysis_data = {}
                    
                    if analysis_json.exists():
                        with open(analysis_json, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                    
                    # Create training item
                    item = {
                        'image_path': str(img_path),
                        'image_id': base_name,
                        'analysis': analysis_data,
                        'cross_regions': {
                            'top': analysis_data.get('regions', {}).get('region_1', {}),
                            'center': analysis_data.get('regions', {}).get('region_2', {}),
                            'bottom': analysis_data.get('regions', {}).get('region_3', {})
                        }
                    }
                    
                    data_items.append(item)
                    
            except Exception as e:
                logger.warning(f"⚠️  Error processing {test_dir}: {e}")
        
        logger.info(f"✅ Loaded {len(data_items)} training samples")
        return Dataset.from_list(data_items)
    
    def collate_fn(self, batch):
        """Custom collate function for cross-convolution features"""
        texts = []
        messages_batch = []

        for item in batch:
            try:
                # Create text prompt with cross-convolution analysis
                prompt = self.create_cross_analysis_prompt(item['analysis'])

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": item['image_path'],
                                "min_pixels": self.min_pixels,
                                "max_pixels": self.max_pixels,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                texts.append(text)
                messages_batch.append(messages)

            except Exception as e:
                logger.warning(f"⚠️  Error processing item: {e}")

        if not texts:
            return None

        image_inputs, video_inputs = process_vision_info(messages_batch)

        # Process with Qwen2-VL
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Use input_ids as labels for causal LM training
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels

        return inputs
    
    def create_cross_analysis_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create training prompt with cross-convolution analysis"""
        regions = analysis.get('regions', {})
        
        prompt = "Analyze this document using cross-convolution structure:\n"
        
        for region_id, region_info in regions.items():
            name = region_info.get('name', 'Unknown')
            dims = region_info.get('dimensions', {})
            coords = region_info.get('coordinates', {})
            
            x_range = coords.get('x', {})
            y_range = coords.get('y', {})
            
            prompt += f"\n{name}:\n"
            prompt += f"  - Position X: [{x_range.get('min', 0)}, {x_range.get('max', 0)}]\n"
            prompt += f"  - Position Y: [{y_range.get('min', 0)}, {y_range.get('max', 0)}]\n"
            prompt += f"  - Size: {dims.get('width', 0)}x{dims.get('height', 0)}\n"
        
        prompt += "\nExtract all text with accurate positioning."
        return prompt
    
    def train(
        self,
        data_dir: str,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5
    ):
        """Fine-tune the model"""
        
        logger.info("🎓 Starting training...")
        
        # Load training data
        train_dataset = self.load_training_data(data_dir)
        
        if len(train_dataset) == 0:
            logger.error("❌ No training data found!")
            return False
        
        # Create training directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            bf16=False,
            gradient_accumulation_steps=1,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.collate_fn,
        )
        
        # Train
        try:
            trainer.train()
            logger.info("✅ Training completed!")
            
            # Save model
            self.model.save_pretrained(os.path.join(output_dir, "final_model"))
            logger.info(f"✅ Model saved to {output_dir}/final_model")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Cross-Convolution Enhanced Training")
    parser.add_argument("--model-path", default="/home/gunhee/blockpass-AI/models/Qwen2-VL-7B-Instruct-KoDocOCR",
                       help="Path to base model")
    parser.add_argument("--init-dir", default="/home/gunhee/blockpass-AI/init",
                       help="Init directory with configs")
    parser.add_argument("--data-dir", default="/home/gunhee/blockpass-AI/Test",
                       help="Training data directory")
    parser.add_argument("--output-dir", default="/home/gunhee/blockpass-AI/init/training_output",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CrossConvolutionTrainer(args.model_path, args.init_dir)
    
    # Load model
    if not trainer.load_model():
        return 1
    
    # Train
    if not trainer.train(
        args.data_dir,
        args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    ):
        return 1
    
    logger.info("\n✅ All done!")
    return 0


if __name__ == "__main__":
    exit(main())
