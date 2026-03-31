"""
Utility Functions for HBU-YOLO-VLM
"""

import torch
import torch.distributed as dist
import os
import random
import numpy as np
from typing import Optional, Dict, Any


def init_distributed() -> bool:
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
        return True
    return False


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if main process"""
    return get_rank() == 0


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_tensor(tensor: torch.Tensor, avg: bool = True) -> torch.Tensor:
    """Reduce tensor across all GPUs"""
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if avg:
        rt /= dist.get_world_size()
    
    return rt


def synchronize():
    """Synchronize all processes"""
    if dist.is_initialized():
        dist.barrier()


def to_cuda(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch to CUDA"""
    cuda_batch = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cuda_batch[key] = value.to(device)
        elif isinstance(value, dict):
            cuda_batch[key] = to_cuda(value, device)
        elif isinstance(value, list):
            cuda_batch[key] = [
                v.to(device) if isinstance(v, torch.Tensor) else v
                for v in value
            ]
        else:
            cuda_batch[key] = value
    
    return cuda_batch


def format_eta(seconds: float) -> str:
    """Format ETA string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_number(num: int) -> str:
    """Format number with commas"""
    return f"{num:,}"


def memory_summary() -> Dict[str, float]:
    """Get GPU memory summary"""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.memory.max_memory_allocated() / 1e9
    }


def print_memory_stats():
    """Print memory statistics"""
    if torch.cuda.is_available():
        print(f"GPU Memory: {memory_summary()}")
