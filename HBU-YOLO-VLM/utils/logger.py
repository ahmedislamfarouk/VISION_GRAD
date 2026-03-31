"""
Logging Utilities for HBU-YOLO-VLM
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch


def setup_logger(
    name: str,
    output_dir: str,
    rank: int = 0,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger
    
    Args:
        name: Logger name
        output_dir: Output directory for logs
        rank: Process rank
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Don't add handlers if already configured
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (only for main process)
    if rank == 0:
        log_file = Path(output_dir) / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class WandBLogger:
    """Weights & Biases logger"""
    
    def __init__(self, config: Any):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            self.wandb = None
            return
        
        # Initialize wandb
        run_name = config.logging.run_name or f"HBU-YOLO-VLM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=dict(config),
            tags=config.logging.get('wandb_tags', []),
            entity=config.logging.get('wandb_entity', None)
        )
        
        self.log_images_flag = config.logging.log_images
    
    def log(self, data: Dict[str, Any], step: int):
        """Log data to wandb"""
        if self.wandb is None:
            return
        
        self.wandb.log(data, step=step)
    
    def log_train_step(self, step: int, metrics: Dict[str, float]):
        """Log training step"""
        if self.wandb is None:
            return
        
        log_data = {f"train/{k}": v for k, v in metrics.items()}
        log_data['step'] = step
        
        self.wandb.log(log_data, step=step)
    
    def log_validation(self, epoch: int, metrics: Dict[str, float]):
        """Log validation metrics"""
        if self.wandb is None:
            return
        
        log_data = {f"val/{k}": v for k, v in metrics.items()}
        log_data['epoch'] = epoch
        
        self.wandb.log(log_data, step=epoch)
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """Log epoch summary"""
        if self.wandb is None:
            return
        
        log_data = {
            'epoch': epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()}
        }
        
        if val_metrics:
            log_data.update({f"val/{k}": v for k, v in val_metrics.items()})
        
        self.wandb.log(log_data, step=epoch)
    
    def log_image(self, image: torch.Tensor, detections: Dict, index: int = 0):
        """Log image with detections"""
        if self.wandb is None or not self.log_images_flag:
            return None
        
        try:
            import wandb
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(image)
            
            # Draw boxes if available
            if 'boxes' in detections:
                boxes = detections['boxes'][index].cpu().numpy()
                for box in boxes[:10]:  # Limit to 10 boxes
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
            
            ax.axis('off')
            plt.tight_layout()
            
            return wandb.Image(fig)
        
        finally:
            plt.close()
    
    def finish(self):
        """Finish wandb run"""
        if self.wandb is not None:
            self.wandb.finish()


class TensorBoardLogger:
    """TensorBoard logger"""
    
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("Warning: tensorboard not installed. Logging disabled.")
            self.writer = None
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log scalar value"""
        if self.writer is None:
            return
        
        self.writer.add_scalar(name, value, step)
    
    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log multiple scalar values"""
        if self.writer is None:
            return
        
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
    
    def log_image(self, name: str, image: torch.Tensor, step: int):
        """Log image"""
        if self.writer is None:
            return
        
        self.writer.add_image(name, image, step)
    
    def close(self):
        """Close writer"""
        if self.writer is not None:
            self.writer.close()
