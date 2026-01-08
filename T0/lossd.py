import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def poincare_distance(x, y, c=1.0, eps=1e-5):
    """
    Compute Poincaré distance between two sets of vectors in hyperbolic space.

    Args:
        x: Tensor of shape (batch_size_1, dim) or (batch_size_1, num_points, dim)
        y: Tensor of shape (batch_size_2, dim) or (batch_size_2, num_points, dim)
        c: Curvature parameter (default: 1.0)
        eps: Small constant for numerical stability

    Returns:
        Distance matrix of shape (batch_size_1, batch_size_2) or averaged distance
    """
    # Ensure inputs are 2D tensors
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)

    # Compute squared norms
    x_norm = torch.sum(x ** 2, dim=-1, keepdim=True)
    y_norm = torch.sum(y ** 2, dim=-1, keepdim=True)

    # Ensure points are within unit ball (Poincaré disk constraint)
    x_norm_clamped = torch.clamp(x_norm, 0, 1 - eps)
    y_norm_clamped = torch.clamp(y_norm, 0, 1 - eps)

    # Compute squared Euclidean distance
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    sq_norm = torch.sum(diff ** 2, dim=-1)

    # Compute Poincaré distance
    # d^2 = (1/c) * arcosh(1 + 2*((x-y)^2) / ((1-|x|^2)*(1-|y|^2)))
    numerator = sq_norm
    denominator = (1 - x_norm_clamped) * (1 - y_norm_clamped).transpose(0, 1)
    denominator = torch.clamp(denominator, min=eps)

    alpha = 2 * numerator / denominator
    alpha = torch.clamp(alpha, min=1 + eps)  # Ensure arcosh argument >= 1

    distance_squared = (1 / c) * torch.arcosh(alpha)
    distance = torch.sqrt(torch.clamp(distance_squared, min=0))

    return distance


def compute_hyperbolic_knn_temperature(teacher_logits, student_logits, k=5, c=1.0):
    """
    Compute temperature factor based on hyperbolic KNN distances.

    This function:
    1. Maps logits to probability distributions using softmax
    2. Computes KNN for each sample using hyperbolic distance
    3. Calculates local density based on distances to k-nearest neighbors
    4. Derives temperature factors inversely proportional to density

    Args:
        teacher_logits: Teacher model logits [batch_size, vocab_size]
        student_logits: Student model logits [batch_size, vocab_size]
        k: Number of nearest neighbors
        c: Hyperbolic curvature parameter

    Returns:
        temperature_t: Temperature factors for teacher [batch_size] or scalar
        temperature_s: Temperature factors for student [batch_size] or scalar
    """
    softmax = nn.Softmax(dim=1)
    p_t = softmax(teacher_logits)
    p_s = softmax(student_logits)

    # Compute pairwise distance matrices using hyperbolic distance
    # For teacher: compute distances between all pairs of teacher samples
    dist_matrix_t = poincare_distance(p_t, p_t, c=c)

    # For student: compute distances between all pairs of student samples
    dist_matrix_s = poincare_distance(p_s, p_s, c=c)

    # For each sample, find k-nearest neighbors (excluding self)
    # Set diagonal to infinity to exclude self from KNN
    batch_size = p_t.size(0)
    diagonal_mask = torch.eye(batch_size, device=p_t.device).bool()

    # Teacher KNN
    dist_matrix_t_masked = dist_matrix_t.clone()
    dist_matrix_t_masked[diagonal_mask] = float('inf')
    knn_dists_t, _ = torch.topk(dist_matrix_t_masked, k=min(k, batch_size-1), largest=False, dim=1)
    avg_knn_dist_t = torch.mean(knn_dists_t, dim=1)  # [batch_size]

    # Student KNN
    dist_matrix_s_masked = dist_matrix_s.clone()
    dist_matrix_s_masked[diagonal_mask] = float('inf')
    knn_dists_s, _ = torch.topk(dist_matrix_s_masked, k=min(k, batch_size-1), largest=False, dim=1)
    avg_knn_dist_s = torch.mean(knn_dists_s, dim=1)  # [batch_size]

    # Compute local density: inversely proportional to average KNN distance
    # Higher density = smaller average distance = higher temperature
    # Add small epsilon for numerical stability
    density_t = 1.0 / (avg_knn_dist_t + 1e-8)
    density_s = 1.0 / (avg_knn_dist_s + 1e-8)

    # Temperature factor is proportional to density
    # Normalize to have mean around 2.0 (similar to original temperature)
    temperature_t = density_t / torch.mean(density_t) * 2.0
    temperature_s = density_s / torch.mean(density_s) * 2.0

    # Return scalar (mean) temperature for simplicity
    # Can also return per-sample temperatures if needed
    return torch.mean(temperature_t), torch.mean(temperature_s)


class KL(nn.Module):
    def __init__(self):
        super(KL, self).__init__()
        self.T = 1
    def forward(self, y_s, y_t, mode="classification", temperature=None):
        """
        Args:
            y_s: Student logits
            y_t: Teacher logits
            mode: "classification" or "regression"
            temperature: Optional temperature scalar to override self.T (for hyperbolic temperature)
        """
        y_s = y_s.view(-1, 32128)
        y_t = y_t.view(-1, 32128)
        y_s = torch.log_softmax(y_s, dim=-1)
        y_t = torch.log_softmax(y_t, dim=-1)

        T = temperature if temperature is not None else self.T

        if mode == "regression":
            loss = F.mse_loss((y_s/T).view(-1), (y_t/T).view(-1))
        else:
            p_s = F.log_softmax(y_s/T, dim=-1)
            p_t = F.softmax(y_t/T, dim=-1)
            loss = -torch.sum(p_t * p_s, dim=-1).mean()
        return loss
    
class Sinkhorn(nn.Module):
    def __init__(self):
        super(Sinkhorn, self).__init__()
        self.T = 2   #0.55 #2
        self.curvature = 1.0  # Hyperbolic curvature parameter

    def sinkhorn_normalized(self,x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=20, use_hyperbolic=True):
        """
        Compute Sinkhorn loss with optional hyperbolic distance metric.

        Args:
            x: Student probability distribution [batch_size, vocab_size]
            y: Teacher probability distribution [batch_size, vocab_size]
            epsilon: Regularization parameter
            n_iters: Number of Sinkhorn iterations
            use_hyperbolic: Whether to use hyperbolic distance (default: True)
        """
        if use_hyperbolic:
            # Use hyperbolic distance metric
            Wxy = poincare_distance(x, y, c=self.curvature)
        else:
            # Use Euclidean distance (original)
            Wxy = torch.cdist(x, y, p=1)

        K = torch.exp(-Wxy / epsilon)  # 计算内核矩阵
        P = self.sinkhorn_normalized(K, n_iters)  # 计算 Sinkhorn 迭代的结果
        return torch.sum(P * Wxy)  # 计算近似 EMD 损失

    def forward(self, y_s, y_t, mode="classification", temperature=None):
        """
        Args:
            y_s: Student logits
            y_t: Teacher logits
            mode: "classification" or "regression"
            temperature: Optional temperature scalar to override self.T (for hyperbolic temperature)
        """
        softmax = nn.Softmax(dim=1)
        # selected_dims = [465,2163]
        # y_s = torch.index_select(y_s, dim=-1, index=torch.tensor(selected_dims).to(y_s.device))
        # y_t = torch.index_select(y_t, dim=-1, index=torch.tensor(selected_dims).to(y_t.device))

        T = temperature if temperature is not None else self.T

        p_s = softmax(y_s/T)
        p_t = softmax(y_t/T)
        emd_loss = 0.0008*self.sinkhorn_loss(x=p_s, y=p_t, use_hyperbolic=True)   # Use hyperbolic distance
        return emd_loss
    
class RKL(nn.Module):
    def __init__(self):
        super(RKL, self).__init__()
        self.T = 2
    def forward(self,  y_s, y_t, mode="classification"):
        temperature = self.T
        if mode == "regression":
            
            loss = F.mse_loss((y_s/temperature).view(-1), (y_t/temperature).view(-1))
        else:
            p_s = F.softmax(y_s/temperature, dim=-1)
            p_s1 = F.log_softmax(y_s/temperature, dim=-1)
            p_t = F.log_softmax(y_t/temperature, dim=-1)
            loss =torch.sum(p_s1 * p_s, dim=-1).mean() -torch.sum(p_t * p_s, dim=-1).mean()
        return 0.1*loss
    
class JSKL(nn.Module):
    def __init__(self):
        super(JSKL, self).__init__()
        self.T = 2
    def js_divergence(self,p, q):
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(p, m, reduction='batchmean') +
                    F.kl_div(q, m, reduction='batchmean'))
    def forward(self, y_s, y_t, mode="classification"):
        temperature = self.T
        if mode == "regression":
            loss = F.mse_loss((y_s/temperature).view(-1), (y_t/temperature).view(-1))
        else:
            p_s = F.softmax(y_s/temperature, dim=-1)
            p_t = F.softmax(y_t/temperature, dim=-1)
            loss = (0.5 * self.js_divergence(p_s, p_t)).mean()
        return 0.1 * loss

