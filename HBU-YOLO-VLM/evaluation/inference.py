"""
Inference Script for HBU-YOLO-VLM
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import cv2

from models.hbu_yolo_vlm import build_model
from datasets.augmentations import build_augmentations


class HBUYOLOVLMPredictor:
    """Predictor for HBU-YOLO-VLM"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load config
        if config_path:
            self.config = OmegaConf.load(config_path)
        else:
            # Try to load config from checkpoint directory
            checkpoint_dir = Path(checkpoint_path).parent
            config_path = checkpoint_dir / "config.yaml"
            if config_path.exists():
                self.config = OmegaConf.load(config_path)
            else:
                raise ValueError("Config file not found")
        
        # Build model
        self.model = build_model(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        # Build augmentations for inference
        self.augmentations = build_augmentations({
            'enabled': True,
            'normalize': True
        })
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict on single image
        
        Args:
            image: Input image (H, W, 3) in RGB format
            prompt: Optional text prompt
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        image_tensor = self._preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model.generate(
            images=image_tensor,
            prompts=[prompt] if prompt else None
        )
        
        # Process output
        predictions = self._postprocess(output, image.shape)
        
        return predictions
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[np.ndarray],
        prompts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict on batch of images
        
        Args:
            images: List of input images
            prompts: Optional list of text prompts
            
        Returns:
            List of predictions
        """
        # Preprocess images
        image_tensors = [self._preprocess(img) for img in images]
        image_batch = torch.stack(image_tensors).to(self.device)
        
        # Forward pass
        output = self.model.generate(
            images=image_batch,
            prompts=prompts
        )
        
        # Process outputs
        predictions = []
        for i, img in enumerate(images):
            img_predictions = self._postprocess(output, img.shape, batch_index=i)
            predictions.append(img_predictions)
        
        return predictions
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image"""
        # Convert to RGB if needed
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Apply augmentations
        augmented = self.augmentations(image=image)
        image_tensor = augmented['image']
        
        return image_tensor
    
    def _postprocess(
        self,
        output: Dict[str, Any],
        image_shape: tuple,
        batch_index: int = 0
    ) -> Dict[str, Any]:
        """Postprocess predictions"""
        predictions = {}
        
        # Get detections
        if 'detections' in output:
            detections = output['detections']
            
            boxes = detections['boxes'][batch_index].cpu().numpy()
            scores = detections['scores'][batch_index].cpu().numpy()
            
            # Filter by confidence
            if scores.ndim == 2:
                max_scores = scores.max(axis=-1)
                labels = scores.argmax(axis=-1)
            else:
                max_scores = scores
                labels = np.zeros_like(scores, dtype=np.int64)
            
            mask = max_scores > self.confidence_threshold
            boxes = boxes[mask]
            scores = max_scores[mask]
            labels = labels[mask]
            
            # Scale boxes to original image size
            h, w = image_shape[:2]
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            
            predictions['boxes'] = boxes
            predictions['scores'] = scores
            predictions['labels'] = labels
        
        # Get uncertainty
        if 'uncertainty' in output:
            predictions['uncertainty'] = output['uncertainty']
        
        # Get generated text
        if 'generated_text' in output:
            predictions['caption'] = output['generated_text'][batch_index]
        
        return predictions
    
    def visualize(
        self,
        image: np.ndarray,
        predictions: Dict[str, Any],
        show_uncertainty: bool = True
    ) -> np.ndarray:
        """Visualize predictions"""
        vis = image.copy()
        
        # Draw boxes
        if 'boxes' in predictions:
            boxes = predictions['boxes']
            scores = predictions['scores']
            labels = predictions['labels']
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Color based on class
                color = self._get_color(label)
                
                # Draw box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{self.model.class_names[label]}: {score:.2f}"
                cv2.putText(vis, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw caption
        if 'caption' in predictions:
            caption = predictions['caption']
            cv2.putText(vis, caption[:50] + "..." if len(caption) > 50 else caption,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def _get_color(self, label: int) -> tuple:
        """Get color for label"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Navy
            (128, 128, 0)   # Olive
        ]
        
        return colors[label % len(colors)]


def main():
    parser = argparse.ArgumentParser("HBU-YOLO-VLM Inference")
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Path to directory of images"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/predictions",
        help="Output directory"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for VLM"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize predictions"
    )
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = HBUYOLOVLMPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single image
    if args.image:
        image = np.array(Image.open(args.image).convert('RGB'))
        
        predictions = predictor.predict(image, prompt=args.prompt)
        
        # Save predictions
        output_path = output_dir / f"{Path(args.image).stem}_predictions.json"
        
        import json
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Visualize
        if args.visualize:
            vis = predictor.visualize(image, predictions)
            vis_path = output_dir / f"{Path(args.image).stem}_visualization.jpg"
            Image.fromarray(vis).save(vis_path)
        
        print(f"Predictions saved to {output_path}")
    
    # Process image directory
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        for image_path in image_paths:
            image = np.array(Image.open(image_path).convert('RGB'))
            
            predictions = predictor.predict(image, prompt=args.prompt)
            
            # Save predictions
            output_path = output_dir / f"{image_path.stem}_predictions.json"
            
            import json
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            # Visualize
            if args.visualize:
                vis = predictor.visualize(image, predictions)
                vis_path = output_dir / f"{image_path.stem}_visualization.jpg"
                Image.fromarray(vis).save(vis_path)
            
            print(f"Processed {image_path.name}")
    
    else:
        print("Please provide either --image or --image-dir")


if __name__ == "__main__":
    main()
