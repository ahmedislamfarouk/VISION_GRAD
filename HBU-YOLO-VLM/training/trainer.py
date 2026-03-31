"""
Trainer Module for HBU-YOLO-VLM

Handles:
- Training loop
- Validation
- Loss computation
- Metrics calculation
- Checkpointing
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from tqdm import tqdm

from utils.metrics import compute_map, compute_detection_metrics, compute_language_metrics
from utils.logger import WandBLogger


class Trainer:
    """Trainer for HBU-YOLO-VLM"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        config: Any,
        logger: Any,
        wandb_logger: Optional[WandBLogger] = None,
        is_distributed: bool = False,
        global_rank: int = 0
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.wandb_logger = wandb_logger
        self.is_distributed = is_distributed
        self.global_rank = global_rank
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Gradient accumulation
        self.accumulation_steps = config.training.gradient_accumulation_steps
        
        # Mixed precision
        self.use_amp = config.training.mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp,
            dtype=getattr(torch, config.training.amp_dtype)
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.output.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.current_epoch = epoch
        
        metrics = {
            'loss': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'lm_loss': 0.0,
            'unc_loss': 0.0
        }
        
        num_batches = len(train_loader)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            disable=self.global_rank != 0,
            total=num_batches
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['images'].to(self.device)
            texts = batch.get('texts', None)
            boxes = batch.get('boxes', None)
            labels = batch.get('labels', None)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=self.scaler.dtype):
                output = self.model(
                    images=images,
                    texts=texts,
                    boxes=boxes,
                    labels=labels,
                    return_loss=True
                )
                
                losses = output['losses']
                loss = losses['total_loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == num_batches - 1:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.clip_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                self.scheduler.step()
            
            # Update metrics
            metrics['loss'] += losses['total_loss'].item()
            for key in ['box_loss', 'cls_loss', 'lm_loss', 'unc_loss']:
                if key in losses:
                    metrics[key] += losses[key].item()
            
            # Average metrics
            avg_metrics = {k: v / (batch_idx + 1) for k, v in metrics.items()}
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{avg_metrics['loss']:.4f}",
                'box': f"{avg_metrics['box_loss']:.4f}",
                'cls': f"{avg_metrics['cls_loss']:.4f}"
            })
            
            # Log to WandB
            if self.wandb_logger and self.global_rank == 0:
                if self.global_step % self.config.logging.log_interval == 0:
                    self.wandb_logger.log_train_step(self.global_step, avg_metrics)
                
                # Log images periodically
                if self.config.logging.log_images and self.global_step % self.config.logging.log_images_interval == 0:
                    self._log_images(images, output, batch_idx)
            
            self.global_step += 1
        
        # Final metrics
        final_metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return final_metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate"""
        self.model.eval()
        
        metrics = {
            'loss': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'lm_loss': 0.0
        }
        
        all_detections = []
        all_ground_truth = []
        all_captions = []
        all_ground_truth_captions = []
        
        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1} [Val]",
            disable=self.global_rank != 0,
            total=len(val_loader)
        )
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            texts = batch.get('texts', None)
            boxes = batch.get('boxes', None)
            labels = batch.get('labels', None)
            
            # Forward pass
            with autocast(enabled=self.use_amp, dtype=self.scaler.dtype):
                output = self.model(
                    images=images,
                    texts=texts,
                    boxes=boxes,
                    labels=labels,
                    return_loss=True
                )
                
                losses = output['losses']
                loss = losses['total_loss']
            
            # Update metrics
            metrics['loss'] += loss.item()
            for key in ['box_loss', 'cls_loss', 'lm_loss']:
                if key in losses:
                    metrics[key] += losses[key].item()
            
            # Collect detections and ground truth
            detections = output['detections']
            all_detections.append(detections)
            
            if boxes is not None:
                all_ground_truth.append({
                    'boxes': boxes.cpu(),
                    'labels': labels.cpu() if labels is not None else None
                })
            
            # Collect captions if available
            if 'generated_text' in output:
                all_captions.extend(output['generated_text'])
            if texts is not None:
                all_ground_truth_captions.extend(texts.get('raw_texts', []))
        
        # Compute metrics
        num_batches = len(val_loader)
        final_metrics = {k: v / num_batches for k, v in metrics.items()}
        
        # Compute mAP
        if all_detections and all_ground_truth:
            map_metrics = compute_map(all_detections, all_ground_truth)
            final_metrics.update(map_metrics)
        
        # Compute language metrics
        if all_captions and all_ground_truth_captions:
            lang_metrics = compute_language_metrics(all_captions, all_ground_truth_captions)
            final_metrics.update(lang_metrics)
        
        # Log to WandB
        if self.wandb_logger and self.global_rank == 0:
            self.wandb_logger.log_validation(epoch, final_metrics)
        
        return final_metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save checkpoint"""
        if self.global_rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "checkpoint_last.pth"
        torch.save(checkpoint, last_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        current_metric = metrics.get(self.config.validation.metric, 0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best checkpoint saved with {self.config.validation.metric}: {current_metric:.4f}")
        
        # Remove old checkpoints
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Checkpoint saved for epoch {epoch + 1}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', 0)
        
        self.logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def _cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Remove old checkpoints"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoints) > keep_last_n:
            # Sort by epoch
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
    
    def _log_images(
        self,
        images: torch.Tensor,
        output: Dict[str, Any],
        batch_idx: int
    ):
        """Log images to WandB"""
        if not self.config.logging.log_images:
            return
        
        # Get detections
        detections = output['detections']
        
        # Log first few images
        num_images = min(4, images.shape[0])
        
        wandb_images = []
        for i in range(num_images):
            # Convert image to numpy
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            
            # Create WandB image
            wandb_img = self.wandb_logger.log_image(
                img,
                detections=detections,
                index=i
            )
            wandb_images.append(wandb_img)
        
        if wandb_images:
            self.wandb_logger.log({"images": wandb_images}, step=self.global_step)
