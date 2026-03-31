"""
Metrics for HBU-YOLO-VLM

Detection Metrics:
- mAP (mean Average Precision)
- Precision, Recall
- IoU

Language Metrics:
- BLEU
- ROUGE
- CIDEr
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocoevalcap.eval import COCOEvalCap
import json
import tempfile


def compute_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: (N, 4) boxes in format [x1, y1, x2, y2]
        boxes2: (M, 4) boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix (N, M)
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    boxes2 = boxes2.unsqueeze(0)  # (1, M, 4)
    
    # Compute intersection
    intersection_min = torch.max(boxes1[:, :, :2], boxes2[:, :, :2])
    intersection_max = torch.min(boxes1[:, :, 2:], boxes2[:, :, 2:])
    
    intersection_wh = (intersection_max - intersection_min).clamp(min=0)
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]
    
    # Compute areas
    boxes1_wh = boxes1[:, :, 2:] - boxes1[:, :, :2]
    boxes2_wh = boxes2[:, :, 2:] - boxes2[:, :, :2]
    
    boxes1_area = boxes1_wh[:, :, 0] * boxes1_wh[:, :, 1]
    boxes2_area = boxes2_wh[:, :, 0] * boxes2_wh[:, :, 1]
    
    # Compute union
    union_area = boxes1_area + boxes2_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / (union_area + 1e-8)
    
    return iou


def compute_ap(
    scores: np.ndarray,
    labels: np.ndarray,
    gt_labels: np.ndarray
) -> float:
    """
    Compute Average Precision
    
    Args:
        scores: Detection scores
        labels: Detection labels
        gt_labels: Ground truth labels
        
    Returns:
        Average Precision
    """
    # Sort by score
    sorted_indices = np.argsort(-scores)
    labels = labels[sorted_indices]
    
    # Compute precision-recall curve
    tp = (labels == gt_labels).astype(float)
    fp = 1 - tp
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (len(gt_labels) + 1e-8)
    
    # Compute AP (area under precision-recall curve)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


def compute_map(
    detections: List[Dict[str, torch.Tensor]],
    ground_truth: List[Dict[str, torch.Tensor]],
    iou_thresholds: List[float] = [0.5, 0.75]
) -> Dict[str, float]:
    """
    Compute mean Average Precision
    
    Args:
        detections: List of detection dictionaries
        ground_truth: List of ground truth dictionaries
        
    Returns:
        Dictionary with mAP metrics
    """
    metrics = {}
    
    # Aggregate all detections and ground truth
    all_detections = []
    all_ground_truth = []
    
    for det, gt in zip(detections, ground_truth):
        boxes = det['boxes'].cpu().numpy()
        scores = det['scores'].cpu().numpy()
        labels = det.get('labels', None)
        
        if labels is None:
            labels = np.argmax(scores, axis=-1)
            scores = np.max(scores, axis=-1)
        else:
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()
        
        gt_boxes = gt['boxes'].cpu().numpy()
        gt_labels = gt.get('labels', None)
        if gt_labels is not None:
            gt_labels = gt_labels.cpu().numpy()
        
        all_detections.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
        
        all_ground_truth.append({
            'boxes': gt_boxes,
            'labels': gt_labels
        })
    
    # Compute mAP at different IoU thresholds
    for iou_thresh in iou_thresholds:
        aps = []
        
        for det, gt in zip(all_detections, all_ground_truth):
            if len(gt['boxes']) == 0:
                continue
            
            # Compute IoU
            iou = compute_iou(
                torch.from_numpy(det['boxes']),
                torch.from_numpy(gt['boxes'])
            ).numpy()
            
            # Match detections
            matched = (iou > iou_thresh).any(axis=1)
            matched_labels = det['labels'][matched]
            
            # Compute AP
            if len(matched_labels) > 0:
                ap = compute_ap(
                    det['scores'][matched],
                    matched_labels,
                    gt['labels']
                )
                aps.append(ap)
        
        if aps:
            metrics[f'map_{int(iou_thresh * 100)}'] = np.mean(aps)
        else:
            metrics[f'map_{int(iou_thresh * 100)}'] = 0.0
    
    # Compute overall mAP
    if 'map_50' in metrics and 'map_75' in metrics:
        metrics['map'] = (metrics['map_50'] + metrics['map_75']) / 2
    elif 'map_50' in metrics:
        metrics['map'] = metrics['map_50']
    
    return metrics


def compute_language_metrics(
    predictions: List[str],
    ground_truth: List[str]
) -> Dict[str, float]:
    """
    Compute language generation metrics
    
    Args:
        predictions: List of predicted captions
        ground_truth: List of ground truth captions
        
    Returns:
        Dictionary with language metrics
    """
    metrics = {}
    
    try:
        # Create temporary COCO-format files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create prediction format
            pred_data = {
                'images': [{'id': i} for i in range(len(predictions))],
                'annotations': [
                    {'image_id': i, 'id': i, 'caption': pred}
                    for i, pred in enumerate(predictions)
                ]
            }
            json.dump(pred_data, f)
            pred_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create ground truth format
            gt_data = {
                'images': [{'id': i} for i in range(len(ground_truth))],
                'annotations': [
                    {'image_id': i, 'id': i, 'caption': gt}
                    for i, gt in enumerate(ground_truth)
                ]
            }
            json.dump(gt_data, f)
            gt_file = f.name
        
        # Load COCO and compute metrics
        coco = COCO(gt_file)
        coco_result = coco.loadRes(pred_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()
        
        # Extract metrics
        metrics = coco_eval.eval
        
    except Exception as e:
        print(f"Error computing language metrics: {e}")
        metrics = {
            'BLEU_4': 0.0,
            'ROUGE_L': 0.0,
            'CIDEr': 0.0
        }
    
    return metrics


def compute_detection_metrics(
    detections: List[Dict[str, torch.Tensor]],
    ground_truth: List[Dict[str, torch.Tensor]],
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute detection metrics (precision, recall, F1)
    
    Args:
        detections: List of detection dictionaries
        ground_truth: List of ground truth dictionaries
        confidence_threshold: Confidence threshold
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for det, gt in zip(detections, ground_truth):
        boxes = det['boxes']
        scores = det['scores']
        
        # Filter by confidence
        if scores.dim() == 2:
            max_scores, labels = scores.max(dim=-1)
        else:
            max_scores = scores
            labels = torch.zeros_like(scores).long()
        
        mask = max_scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        gt_boxes = gt['boxes']
        gt_labels = gt.get('labels', None)
        
        # Compute IoU
        if len(boxes) > 0 and len(gt_boxes) > 0:
            iou = compute_iou(boxes, gt_boxes)
            
            # Match detections
            matched = iou.max(dim=1)[0] > 0.5
            
            tp = matched.sum().item()
            fp = len(boxes) - tp
            fn = len(gt_boxes) - matched.sum().item()
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
        else:
            if len(boxes) > 0:
                fp_total += len(boxes)
            if len(gt_boxes) > 0:
                fn_total += len(gt_boxes)
    
    # Compute precision, recall, F1
    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    return metrics
