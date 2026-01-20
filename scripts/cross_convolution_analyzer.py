"""
Cross-Shape Convolution Analyzer for OCR
Analyzes photos in a cross shape structure with colored visualization boxes
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossConvolutionAnalyzer:
    """
    Analyzes OCR documents using cross-shape convolution
    Three photos cross each other in a cross pattern for context analysis
    """

    def __init__(self):
        self.box_colors = {
            'top': (255, 0, 0),       # Blue
            'left': (0, 255, 0),      # Green
            'center': (0, 0, 255),    # Red
            'right': (255, 255, 0),   # Cyan
            'bottom': (255, 0, 255)   # Magenta
        }
        self.box_thickness = 3

    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        return max(min_val, min(value, max_val))

    def _make_box(self, cx: int, cy: int, w: int, h: int, width: int, height: int) -> Tuple[int, int, int, int]:
        x1 = self._clamp(cx - w // 2, 0, width - 1)
        y1 = self._clamp(cy - h // 2, 0, height - 1)
        x2 = self._clamp(cx + w // 2, 1, width)
        y2 = self._clamp(cy + h // 2, 1, height)
        return x1, y1, x2, y2

    def _get_visual_boxes(self, width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Return overlapping cross-style boxes centered with strong overlap."""
        center_x = width // 2
        center_y = height // 2

        offset_x = int(width * 0.12)
        offset_y = int(height * 0.12)

        boxes = {
            'top': self._make_box(center_x, center_y - offset_y, int(width * 0.75), int(height * 0.35), width, height),
            'left': self._make_box(center_x - offset_x, center_y, int(width * 0.35), int(height * 0.75), width, height),
            'center': self._make_box(center_x, center_y, int(width * 0.60), int(height * 0.60), width, height),
            'right': self._make_box(center_x + offset_x, center_y, int(width * 0.35), int(height * 0.75), width, height),
            'bottom': self._make_box(center_x, center_y + offset_y, int(width * 0.75), int(height * 0.35), width, height),
        }
        return boxes

    def _get_analysis_regions(self, width: int, height: int) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """Return three overlapping regions with distinct aspect ratios."""
        center_x = width // 2
        center_y = height // 2
        offset_y = int(height * 0.12)

        regions = {
            'region_1': {
                'name': 'Wide Upper Region',
                'box': self._make_box(center_x, center_y - offset_y, int(width * 0.80), int(height * 0.35), width, height),
            },
            'region_2': {
                'name': 'Square Center Region',
                'box': self._make_box(center_x, center_y, int(width * 0.55), int(height * 0.55), width, height),
            },
            'region_3': {
                'name': 'Tall Lower Region',
                'box': self._make_box(center_x, center_y + offset_y, int(width * 0.35), int(height * 0.80), width, height),
            },
        }
        return regions

    def analyze_xy_values(
        self,
        image: np.ndarray,
        extracted_text: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze x and y coordinates in square structure
        Structure: (1, x), (x, y), (y, small_x), (small_y, long_value)
        """
        height, width = image.shape[:2]
        
        # Define three overlapping regions with distinct directions
        regions = self._get_analysis_regions(width, height)
        
        analysis = {
            'image_dimensions': {'width': width, 'height': height},
            'regions': {},
            'extracted_text': extracted_text
        }
        
        for region_id, region_info in regions.items():
            x_min, y_min, x_max, y_max = region_info['box']
            
            analysis['regions'][region_id] = {
                'name': region_info['name'],
                'coordinates': {
                    'x': {'min': x_min, 'max': x_max, 'range': x_max - x_min},
                    'y': {'min': y_min, 'max': y_max, 'range': y_max - y_min}
                },
                'dimensions': {
                    'width': x_max - x_min,
                    'height': y_max - y_min
                }
            }
        
        return analysis

    def create_cross_visualization(
        self,
        image: np.ndarray,
        extracted_text: Dict[str, Any],
        output_path: str
    ) -> np.ndarray:
        """
        Create visualization with colored boxes in cross pattern
        
        Visual structure:
              [TOP]
        [LEFT][CENTER][RIGHT]
             [BOTTOM]
        """
        vis_image = image.copy()
        height, width = image.shape[:2]
        
        # Define overlapping regions with different colors
        boxes = self._get_visual_boxes(width, height)

        regions_to_draw = [
            {
                'name': 'TOP',
                'box': boxes['top'],
                'color': self.box_colors['top'],
                'label': 'TOP'
            },
            {
                'name': 'LEFT',
                'box': boxes['left'],
                'color': self.box_colors['left'],
                'label': 'LEFT'
            },
            {
                'name': 'CENTER',
                'box': boxes['center'],
                'color': self.box_colors['center'],
                'label': 'CENTER'
            },
            {
                'name': 'RIGHT',
                'box': boxes['right'],
                'color': self.box_colors['right'],
                'label': 'RIGHT'
            },
            {
                'name': 'BOTTOM',
                'box': boxes['bottom'],
                'color': self.box_colors['bottom'],
                'label': 'BOTTOM'
            }
        ]
        
        # Draw boxes
        for region in regions_to_draw:
            x1, y1, x2, y2 = region['box']
            cv2.rectangle(
                vis_image,
                (x1, y1),
                (x2, y2),
                region['color'],
                self.box_thickness
            )
            
            # Add label
            cv2.putText(
                vis_image,
                region['label'],
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                region['color'],
                2
            )
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        logger.info(f"✅ Visualization saved: {output_path}")
        
        return vis_image

    def create_convolution_mask(
        self,
        image: np.ndarray,
        output_path: str
    ) -> np.ndarray:
        """
        Create cross-pattern convolution mask
        Shows overlapping regions for context analysis
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        boxes = self._get_visual_boxes(width, height)
        
        # Draw translucent overlays for cross pattern
        overlay = mask.copy()
        
        # Top
        cv2.rectangle(overlay, (boxes['top'][0], boxes['top'][1]), (boxes['top'][2], boxes['top'][3]),
                     self.box_colors['top'], -1)
        # Left
        cv2.rectangle(overlay, (boxes['left'][0], boxes['left'][1]), (boxes['left'][2], boxes['left'][3]),
                     self.box_colors['left'], -1)
        # Center
        cv2.rectangle(overlay, (boxes['center'][0], boxes['center'][1]), (boxes['center'][2], boxes['center'][3]),
                     self.box_colors['center'], -1)
        # Right
        cv2.rectangle(overlay, (boxes['right'][0], boxes['right'][1]), (boxes['right'][2], boxes['right'][3]),
                     self.box_colors['right'], -1)
        # Bottom
        cv2.rectangle(overlay, (boxes['bottom'][0], boxes['bottom'][1]), (boxes['bottom'][2], boxes['bottom'][3]),
                     self.box_colors['bottom'], -1)
        
        # Blend
        mask = cv2.addWeighted(overlay, 0.4, mask, 0.6, 0)
        cv2.imwrite(output_path, mask)
        logger.info(f"✅ Convolution mask saved: {output_path}")
        
        return mask


def process_and_visualize_image(
    image_path: str,
    json_path: str,
    output_dir: str,
    analyzer: CrossConvolutionAnalyzer
) -> bool:
    """Process a single image and create visualizations"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Failed to read image: {image_path}")
            return False
        
        # Read JSON label data
        json_data = {}
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        
        # Create output subdirectories
        source_dir = os.path.join(output_dir, "Source Photo")
        visualized_dir = os.path.join(output_dir, "Boxed and Visualized Photo")
        json_dir = os.path.join(output_dir, "..", "json")
        
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(visualized_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        
        # Get base filename
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Copy original image
        source_output = os.path.join(source_dir, f"{base_filename}.jpg")
        cv2.imwrite(source_output, image)
        logger.info(f"✅ Source image saved: {source_output}")
        
        # Create visualized image
        vis_path = os.path.join(visualized_dir, f"{base_filename}_visualized.jpg")
        analyzer.create_cross_visualization(image, json_data, vis_path)
        
        # Create convolution mask
        mask_path = os.path.join(visualized_dir, f"{base_filename}_convolution_mask.png")
        analyzer.create_convolution_mask(image, mask_path)
        
        # Save analysis
        analysis = analyzer.analyze_xy_values(image, json_data)
        json_output = os.path.join(json_dir, f"{base_filename}_analysis.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Analysis saved: {json_output}")

        # Copy original label data for reference
        if json_path and os.path.exists(json_path):
            label_output = os.path.join(json_dir, f"{base_filename}_label.json")
            shutil.copy2(json_path, label_output)
            logger.info(f"✅ Label copied: {label_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing image {image_path}: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    analyzer = CrossConvolutionAnalyzer()
    logger.info("🔄 Cross-Convolution Analyzer initialized")
