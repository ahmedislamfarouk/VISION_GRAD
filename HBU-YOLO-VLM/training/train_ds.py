"""
DeepSpeed Training Script for HBU-YOLO-VLM

Optimized for 8×A6000 GPUs with ZeRO-2 optimization
"""

import argparse
import torch
from pathlib import Path
import deepspeed
from omegaconf import OmegaConf

from training.train import TrainingRunner


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="HBU-YOLO-VLM DeepSpeed Training")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hbu_yolo_vlm_full.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory"
    )
    
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Enable DeepSpeed"
    )
    
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="configs/deepspeed_config.json",
        help="DeepSpeed configuration file"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Enable DeepSpeed in config
    config.distributed.deepspeed.enabled = True
    config.distributed.deepspeed.config_file = args.deepspeed_config
    
    # Create output directory
    output_dir = args.output_dir or config.output.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize DeepSpeed
    runner = TrainingRunner(config, args)
    
    # Wrap optimizer with DeepSpeed
    if args.deepspeed:
        runner.model, runner.optimizer, _, runner.scheduler = deepspeed.initialize(
            model=runner.model,
            optimizer=runner.optimizer,
            config=args.deepspeed_config
        )
    
    # Run training
    runner.train()


if __name__ == "__main__":
    main()
