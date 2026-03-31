"""
HBU-YOLO-VLM Training Script

Supports:
- Single GPU training
- Multi-GPU DDP training (optimized for 8×A6000)
- DeepSpeed ZeRO training
- Mixed precision (FP16/BF16)
- Gradient checkpointing
- LoRA fine-tuning
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from omegaconf import OmegaConf, DictConfig
import hydra
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hbu_yolo_vlm import build_model
from datasets.disaster_dataset import build_dataset, collate_fn
from training.trainer import Trainer
from utils.logger import setup_logger, WandBLogger
from utils.distributed import init_distributed, cleanup_distributed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="HBU-YOLO-VLM Training")
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hbu_yolo_vlm_base.yaml",
        help="Path to config file"
    )
    
    # Override config options
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override config options"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs"
    )
    
    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    # Pretrained
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Load pretrained model"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode"
    )
    
    # Local rank
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


class TrainingRunner:
    """Main training runner"""
    
    def __init__(self, config: DictConfig, args):
        self.config = config
        self.args = args
        
        # Initialize distributed training
        self.is_distributed = init_distributed()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.global_rank = int(os.environ.get("RANK", 0))
        
        # Set device
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Setup logger
        self.logger = setup_logger(
            name="HBU-YOLO-VLM",
            output_dir=config.output.output_dir,
            rank=self.global_rank
        )
        
        # Setup WandB
        if self.global_rank == 0 and config.logging.wandb:
            self.wandb_logger = WandBLogger(config)
        else:
            self.wandb_logger = None
        
        # Build model
        self.logger.info("Building model...")
        self.model = build_model(config)
        self.model.to(self.device)
        
        # Apply gradient checkpointing
        if config.training.gradient_checkpointing:
            self.model.apply(self._enable_gradient_checkpointing)
            self.logger.info("Gradient checkpointing enabled")
        
        # Wrap with DDP
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=config.distributed.find_unused_parameters,
                broadcast_buffers=config.distributed.broadcast_buffers
            )
        
        # Build datasets
        self.logger.info("Building datasets...")
        self.train_dataset = build_dataset(
            config=config.dataset,
            data_dir=config.data.train_data_dir,
            split="train",
            augmentations=config.dataset.augmentations
        )
        self.val_dataset = build_dataset(
            config=config.dataset,
            data_dir=config.data.val_data_dir,
            split="val",
            augmentations=None
        )
        
        # Build dataloaders
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Build optimizer
        self.optimizer = self._build_optimizer()
        
        # Build scheduler
        self.scheduler = self._build_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler(
            enabled=config.training.mixed_precision,
            dtype=getattr(torch, config.training.amp_dtype)
        )
        
        # Build trainer
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            config=config,
            logger=self.logger,
            wandb_logger=self.wandb_logger,
            is_distributed=self.is_distributed,
            global_rank=self.global_rank
        )
        
        # Resume or load pretrained
        if args.resume:
            self._load_checkpoint(args.resume)
        elif args.pretrained:
            self._load_pretrained(args.pretrained)
        
        self.logger.info(f"Model parameters: {self._count_parameters():,}")
        self.logger.info(f"Training on {self.world_size} GPU(s)")
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer"""
        config = self.config.training
        
        # Separate parameters for different learning rates
        param_groups = []
        
        # Backbone parameters (lower LR)
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "yolo_backbone" in name or "vision_encoder" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": config.base_lr * 0.1,  # Lower LR for backbone
                "weight_decay": config.weight_decay
            })
        
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": config.base_lr,
                "weight_decay": config.weight_decay
            })
        
        # Build optimizer
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.base_lr,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=config.base_lr,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=config.base_lr,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        return optimizer
    
    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler"""
        config = self.config.training
        
        num_training_steps = len(self.train_loader) * config.num_epochs
        num_warmup_steps = len(self.train_loader) * config.warmup_epochs
        
        if config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=config.lr_min
            )
        elif config.lr_scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.lr_min / config.base_lr,
                total_iters=num_training_steps - num_warmup_steps
            )
        elif config.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")
        
        return scheduler
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.trainer.load_checkpoint(checkpoint_path)
    
    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained model"""
        self.logger.info(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        
        # Load state dict
        self.model.load_state_dict(checkpoint, strict=False)
        self.logger.info("Pretrained model loaded")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _enable_gradient_checkpointing(self, module):
        """Enable gradient checkpointing for module"""
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
    
    def train(self):
        """Run training"""
        self.logger.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            self.logger.info(f"{'='*50}\n")
            
            # Set epoch for sampler
            if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train for one epoch
            train_metrics = self.trainer.train_one_epoch(
                train_loader=self.train_loader,
                epoch=epoch
            )
            
            # Validate
            if (epoch + 1) % self.config.validation.val_interval == 0:
                val_metrics = self.trainer.validate(
                    val_loader=self.val_loader,
                    epoch=epoch
                )
                
                # Save checkpoint
                self.trainer.save_checkpoint(
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=val_metrics[self.config.validation.metric]
                )
            
            # Log epoch summary
            self.logger.info(f"\nEpoch {epoch + 1} Summary:")
            self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            if (epoch + 1) % self.config.validation.val_interval == 0:
                self.logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
                self.logger.info(f"  Val mAP: {val_metrics.get('map', 0):.4f}")
            
            # Log to WandB
            if self.wandb_logger:
                self.wandb_logger.log_epoch(epoch, train_metrics, val_metrics if (epoch + 1) % self.config.validation.val_interval == 0 else None)
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f"\nTraining completed in {total_time_str}")
        
        # Cleanup
        if self.is_distributed:
            cleanup_distributed()


@hydra.main(config_path="configs", config_name="hbu_yolo_vlm_base", version_base=None)
def main_hydra(config: DictConfig):
    """Main function with Hydra config"""
    args = parse_args()
    
    # Override config with command line args
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    # Create output directory
    Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    OmegaConf.save(config, Path(config.output.output_dir) / "config.yaml")
    
    # Run training
    runner = TrainingRunner(config, args)
    runner.train()


if __name__ == "__main__":
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override with command line
    if args.opts:
        config_cli = OmegaConf.from_dotlist(args.opts)
        config = OmegaConf.merge(config, config_cli)
    
    # Create output directory
    output_dir = args.output_dir or config.output.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    OmegaConf.save(config, Path(output_dir) / "config.yaml")
    
    # Run training
    runner = TrainingRunner(config, args)
    runner.train()
