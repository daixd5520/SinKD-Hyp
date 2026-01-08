"""
Geometry Experts for MoG-SKD

Implements three geometric experts:
1. Fisher-Rao Expert (Information Geometry)
2. Euclidean Expert (Baseline)
3. Hyperbolic Expert (Rigorous Hyperbolic Geometry)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryExpert(nn.Module):
    """
    Base class for all geometry experts.

    All experts take logits as input and output a scalar loss for each sample.
    """
    def forward(self, z_s, z_t, **kwargs):
        """
        Args:
            z_s: Student logits [batch_size, num_classes]
            z_t: Teacher logits [batch_size, num_classes]

        Returns:
            loss: Per-sample loss [batch_size]
        """
        raise NotImplementedError("Subclasses must implement forward")


# ==========================================
# Expert A: Fisher-Rao / Hellinger (Information Geometry)
# ==========================================
class FisherRaoExpert(GeometryExpert):
    """
    KDD Selling Point:
    The 'legitimate' geometry of probability distributions.
    Parameter-free, numerically extremely stable.
    Essentially minimizes Hellinger Distance.

    Mathematical Background:
    - Maps probabilities to sphere via sqrt transformation
    - Uses Bhattacharyya coefficient (spherical inner product)
    - Loss = 1 - rho (equivalent to minimizing Hellinger distance)
    """
    def __init__(self, T=1.0):
        super().__init__()
        self.T = T

    def forward(self, z_s, z_t):
        """
        Args:
            z_s: Student logits [batch_size, num_classes]
            z_t: Teacher logits [batch_size, num_classes]

        Returns:
            loss: Per-sample Hellinger distance [batch_size]
        """
        # 1. Soften distributions with temperature
        p_s = F.softmax(z_s / self.T, dim=-1)
        p_t = F.softmax(z_t / self.T, dim=-1)

        # 2. Map to sphere (Sqrt map for information geometry)
        sqrt_p_s = torch.sqrt(p_s + 1e-8)
        sqrt_p_t = torch.sqrt(p_t + 1e-8)

        # 3. Bhattacharyya Coefficient (spherical inner product)
        # rho = <sqrt(p_s), sqrt(p_t)> in L2 space
        rho = torch.sum(sqrt_p_s * sqrt_p_t, dim=-1)

        # 4. Loss: 1 - rho (equivalent to minimizing Hellinger distance)
        # H^2(p, q) = 2 * (1 - BC(p, q))
        # We avoid direct arccos because gradients explode when rho -> 1
        loss = 1.0 - rho

        return loss


# ==========================================
# Expert B: Euclidean Sinkhorn (Strong Baseline)
# ==========================================
class EuclideanExpert(GeometryExpert):
    """
    KDD Selling Point:
    Classic Optimal Transport distillation, serving as a robust baseline.
    Uses L2 ground cost + Sinkhorn algorithm.

    Note: When no predefined Cost Matrix is available,
    we use MSE loss on logits as the Euclidean representative.
    """
    def __init__(self, T=1.0, sinkhorn_eps=0.1, n_iters=10, use_sinkhorn=False):
        super().__init__()
        self.T = T
        self.eps = sinkhorn_eps
        self.n_iters = n_iters
        self.use_sinkhorn = use_sinkhorn

    def _sinkhorn_loss(self, p_s, p_t):
        """
        Compute Sinkhorn divergence using L1 ground metric.

        Args:
            p_s: Student probs [batch_size, num_classes]
            p_t: Teacher probs [batch_size, num_classes]

        Returns:
            loss: Per-sample loss [batch_size]
        """
        batch_size = p_s.size(0)

        # Compute pairwise cost matrix using L1 distance
        # Cost[i, j] = |p_s[i] - p_t[j]|_1
        p_s_expanded = p_s.unsqueeze(1)  # [batch, 1, num_classes]
        p_t_expanded = p_t.unsqueeze(0)  # [1, batch, num_classes]

        # L1 distance matrix
        C = torch.abs(p_s_expanded - p_t_expanded).sum(dim=-1)  # [batch, batch]

        # Kernel matrix
        K = torch.exp(-C / self.eps)

        # Sinkhorn iterations (normalized to keep sum = 1)
        for _ in range(self.n_iters):
            # Row normalization
            K = K / K.sum(dim=1, keepdim=True).clamp_min(1e-8)
            # Column normalization
            K = K / K.sum(dim=0, keepdim=True).clamp_min(1e-8)

        # Optimal transport plan
        P = K

        # Sinkhorn loss = sum(P * C) / batch_size
        # We want diagonal to have high mass (i.e., match corresponding samples)
        loss = torch.sum(P * C, dim=1) / batch_size

        return loss

    def forward(self, z_s, z_t):
        """
        Args:
            z_s: Student logits [batch_size, num_classes]
            z_t: Teacher logits [batch_size, num_classes]

        Returns:
            loss: Per-sample loss [batch_size]
        """
        p_s = F.softmax(z_s / self.T, dim=-1)
        p_t = F.softmax(z_t / self.T, dim=-1)

        if self.use_sinkhorn:
            # Use Sinkhorn divergence
            loss = self._sinkhorn_loss(p_s, p_t)
        else:
            # Use simple MSE on logits (more stable, faster)
            # This represents Euclidean geometry in logit space
            loss = F.mse_loss(z_s, z_t, reduction='none').mean(dim=-1)

        return loss


# ==========================================
# Expert C: Rigorous Hyperbolic (严谨双曲)
# ==========================================
class HyperbolicExpert(GeometryExpert):
    """
    KDD Selling Point:
    Rigorous Log-odds -> Tangent -> Manifold mapping.
    Fixes the geometric error of directly mapping softmax outputs.

    Key Innovation:
    - Treats logits as log-odds (natural coordinates in R^n)
    - Centers them to get tangent vectors at origin of hyperbolic space
    - Uses exponential map to project to hyperbolic manifold
    - Computes Lorentz distance (hyperbolic geometry)
    """
    def __init__(self, T=1.0, c=1.0, learnable_curvature=True):
        super().__init__()
        self.T = T
        if learnable_curvature:
            self.c = nn.Parameter(torch.tensor([c]), requires_grad=True)
        else:
            self.register_buffer('c', torch.tensor([c]))

    def _log_odds_map(self, logits):
        """
        Key Step: Treat logits as log-odds, center them as tangent vectors.

        Log-odds are defined on R^n and can serve as tangent space T_0 H.

        Mathematical Justification:
        - Log-odds: log(p / (1-p)) or equivalently log(p) - constant
        - Centering: removes the constraint sum(p) = 1
        - Result: vectors in unconstrained R^n, isometric to Euclidean subspace

        Args:
            logits: [batch_size, num_classes]

        Returns:
            tangent_vec: [batch_size, num_classes]
        """
        probs = F.softmax(logits / self.T, dim=-1)

        # Log-odds with centering (Aitchison geometry for compositional data)
        log_probs = torch.log(probs + 1e-9)

        # Centering: make sum zero (tangent space at origin)
        tangent_vec = log_probs - log_probs.mean(dim=-1, keepdim=True)

        return tangent_vec

    def _exp_map(self, v):
        """
        Exponential Map: Tangent -> Hyperbolic (Lorentz Model)

        Maps tangent vector v at origin to point on hyperbolic manifold.

        Lorentz Model:
        - Points: x = (x0, x_rest) where -x0^2 + ||x_rest||^2 = -1/c
        - Metric: <x, y>_L = -x0*y0 + <x_rest, y_rest>

        Args:
            v: Tangent vectors [batch_size, num_classes]

        Returns:
            x: Hyperbolic points [batch_size, num_classes+1]
        """
        sqrt_c = torch.sqrt(torch.abs(self.c) + 1e-5)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)

        # Avoid division by zero and numerical issues
        v_norm_scaled = v_norm * sqrt_c

        # Lorentz exponential map formula
        # x0 = cosh(|v|*sqrt(c)) / sqrt(c)
        # x1..n = sinh(|v|*sqrt(c)) * v / (|v|*sqrt(c))

        x0 = torch.cosh(v_norm_scaled) / sqrt_c
        x_rest = torch.sinh(v_norm_scaled) * (v / v_norm_scaled)

        # Concatenate: [batch, 1] + [batch, num_classes] = [batch, num_classes+1]
        x = torch.cat([x0, x_rest], dim=-1)

        return x

    def _lorentz_dist(self, x, y):
        """
        Lorentz (Hyperbolic) Distance between two points on manifold.

        Uses Lorentz inner product and acosh for distance computation.

        Args:
            x: Hyperbolic points [batch_size, dim+1]
            y: Hyperbolic points [batch_size, dim+1]

        Returns:
            dist: Hyperbolic distances [batch_size]
        """
        # Split into time and space components
        x0, x_rest = x[..., :1], x[..., 1:]
        y0, y_rest = y[..., :1], y[..., 1:]

        # Lorentz inner product: <x, y>_L = -x0*y0 + <x_rest, y_rest>
        inner = -x0 * y0 + (x_rest * y_rest).sum(dim=-1, keepdim=True)

        # Hyperbolic distance formula
        # d(x, y) = acosh(-c * <x, y>_L) / sqrt(c)
        sqrt_c = torch.sqrt(torch.abs(self.c) + 1e-5)

        # Clamp argument for numerical stability
        # acosh requires argument >= 1, we add small epsilon
        dist_arg = (-inner * self.c).clamp_min(1.0 + 1e-6)

        # Compute distance
        dist = torch.acosh(dist_arg) / sqrt_c

        return dist.squeeze(-1)

    def forward(self, z_s, z_t):
        """
        Args:
            z_s: Student logits [batch_size, num_classes]
            z_t: Teacher logits [batch_size, num_classes]

        Returns:
            loss: Hyperbolic distance [batch_size]
        """
        # 1. Logits -> Tangent space vectors (log-odds map)
        v_s = self._log_odds_map(z_s)
        v_t = self._log_odds_map(z_t)

        # 2. Tangent space -> Hyperbolic manifold points (exponential map)
        h_s = self._exp_map(v_s)
        h_t = self._exp_map(v_t)

        # 3. Compute hyperbolic (Lorentz) distance
        dist = self._lorentz_dist(h_s, h_t)

        return dist
