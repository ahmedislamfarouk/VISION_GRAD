"""
Evaluation Script for HBU-YOLO-VLM
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf
from tqdm import tqdm

from models.hbu_yolo_vlm import build_model
from datasets.disaster_dataset import build_dataset, build_dataloader, collate_fn
from utils.metrics import compute_map, compute_detection_metrics, compute_language_metrics


class Evaluator:
    """Evaluator for HBU-YOLO-VLM"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda"
    ):
        self.device = device
        
        # Load config
        self.config = OmegaConf.load(config_path)
        
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
    
    @torch.no_grad()
    def evaluate(
        self,
        data_dir: str,
        split: str = "test",
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset
        
        Args:
            data_dir: Dataset directory
            split: Data split
            batch_size: Batch size
            
        Returns:
            Dictionary with all metrics
        """
        # Build dataset
        dataset = build_dataset(
            config=self.config.dataset,
            data_dir=data_dir,
            split=split,
            augmentations=None
        )
        
        # Build dataloader
        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            distributed=False
        )
        
        # Evaluation metrics
        all_detections = []
        all_ground_truth = []
        all_captions = []
        all_ground_truth_captions = []
        
        metrics = {
            'loss': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'lm_loss': 0.0
        }
        
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            boxes = batch.get('boxes', None)
            labels = batch.get('labels', None)
            texts = batch.get('texts', None)
            
            # Forward pass
            output = self.model(
                images=images,
                texts=texts,
                boxes=boxes,
                labels=labels,
                return_loss=True
            )
            
            # Compute loss
            losses = output['losses']
            for key in metrics:
                if key in losses:
                    metrics[key] += losses[key].item()
            
            # Collect detections
            detections = output['detections']
            all_detections.append(detections)
            
            # Collect ground truth
            if boxes is not None:
                all_ground_truth.append({
                    'boxes': boxes,
                    'labels': labels
                })
            
            # Collect captions
            if 'generated_text' in output:
                all_captions.extend(output['generated_text'])
            if texts is not None:
                all_ground_truth_captions.extend(texts.get('raw_texts', []))
        
        # Average losses
        num_batches = len(dataloader)
        for key in metrics:
            metrics[key] /= num_batches
        
        # Compute mAP
        map_metrics = compute_map(all_detections, all_ground_truth)
        metrics.update(map_metrics)
        
        # Compute detection metrics
        det_metrics = compute_detection_metrics(all_detections, all_ground_truth)
        metrics.update(det_metrics)
        
        # Compute language metrics
        if all_captions and all_ground_truth_captions:
            lang_metrics = compute_language_metrics(all_captions, all_ground_truth_captions)
            metrics.update(lang_metrics)
        
        return metrics


def main():
    parser = argparse.ArgumentParser("HBU-YOLO-VLM Evaluation")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset directory"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Evaluate
    metrics = evaluator.evaluate(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    for key, value in sorted(metrics.items()):
        print(f"{key:20s}: {value:.4f}")
    
    print("="*50)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    output_path = output_dir / f"evaluation_{args.split}.json"
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
