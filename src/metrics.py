import torch
import torch.nn.functional as F

def critical_success_index(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    CSI = Hits / (Hits + Misses + False Alarms)
    Used heavily in meteorology for precipitation evaluation at a specific threshold.
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    hits = torch.sum(pred_bin * target_bin)
    misses = torch.sum((1 - pred_bin) * target_bin)
    false_alarms = torch.sum(pred_bin * (1 - target_bin))
    csi = hits / (hits + misses + false_alarms + 1e-8)
    return csi.item()

def structural_similarity_index(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    Calculates SSIM.
    Useful for meteorology to check if the spatial structure of storms/cyclones is correctly predicted,
    even if the exact intensity is slightly off.
    Assumes dimensions are NCHW or Sequence NCHW (flattens out batch and sequence).
    """
    # Flatten batch and sequence dimensions for SSIM over images
    if pred.dim() == 5: # [B, S, C, H, W]
        b, s, c, h, w = pred.shape
        pred = pred.view(b * s, c, h, w)
        target = target.view(b * s, c, h, w)
        
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM requires channel-wise calculation
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()
