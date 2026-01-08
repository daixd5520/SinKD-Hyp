"""
Sinkhorn Solver for Optimal Transport

Implements Sinkhorn algorithm for entropy-regularized optimal transport.
Used by EuclideanExpert and potentially other experts.
"""

import torch
import torch.nn as nn


class SinkhornSolver(nn.Module):
    """
    Sinkhorn Algorithm for Entropy-Regularized Optimal Transport.

    Solves:
        min_P <P, C> + epsilon * H(P)
        s.t. P 1 = a, P^T 1 = b

    where:
        - C is cost matrix
        - H(P) is entropy of P
        - a, b are marginal distributions
        - epsilon is regularization parameter
    """
    def __init__(self, epsilon=0.1, n_iters=10, convergence_tol=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.convergence_tol = convergence_tol

    def sinkhorn_knopp(self, C, a=None, b=None):
        """
        Sinkhorn-Knopp algorithm with log-domain stabilization.

        Args:
            C: Cost matrix [batch_size, batch_size]
            a: Source marginals [batch_size] (default: uniform)
            b: Target marginals [batch_size] (default: uniform)

        Returns:
            P: Transport plan [batch_size, batch_size]
        """
        batch_size = C.size(0)

        # Default to uniform marginals
        if a is None:
            a = torch.ones(batch_size, device=C.device) / batch_size
        if b is None:
            b = torch.ones(batch_size, device=C.device) / batch_size

        # Kernel matrix: K = exp(-C / epsilon)
        K = torch.exp(-C / self.epsilon)

        # Initialize scaling factors
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        # Sinkhorn iterations
        for i in range(self.n_iters):
            u_prev = u.clone()

            # Update u
            u = a / (K @ v)

            # Update v
            v = b / (K.T @ u)

            # Check convergence
            if torch.norm(u - u_prev) < self.convergence_tol:
                break

        # Construct transport plan
        P = torch.diag(u) @ K @ torch.diag(v)

        return P

    def sinkhorn_divergence(self, C, a=None, b=None):
        """
        Compute Sinkhorn divergence (transport cost + entropy).

        Args:
            C: Cost matrix [batch_size, batch_size]
            a: Source marginals [batch_size]
            b: Target marginals [batch_size]

        Returns:
            divergence: Scalar value
            P: Transport plan [batch_size, batch_size]
        """
        P = self.sinkhorn_knopp(C, a, b)

        # Transport cost: <P, C>
        transport_cost = torch.sum(P * C)

        # Entropy regularization: epsilon * H(P)
        entropy = -torch.sum(P * torch.log(P + 1e-9))
        regularization = self.epsilon * entropy

        divergence = transport_cost + regularization

        return divergence, P

    def forward(self, x, y, cost_metric='l2'):
        """
        Compute Sinkhorn divergence between two sets of vectors.

        Args:
            x: Source vectors [batch_size, dim]
            y: Target vectors [batch_size, dim]
            cost_metric: 'l1' or 'l2'

        Returns:
            divergence: Scalar Sinkhorn divergence
            P: Transport plan [batch_size, batch_size]
        """
        # Compute cost matrix
        if cost_metric == 'l1':
            C = torch.cdist(x, y, p=1)
        elif cost_metric == 'l2':
            C = torch.cdist(x, y, p=2)
        else:
            raise ValueError(f"Unknown cost metric: {cost_metric}")

        divergence, P = self.sinkhorn_divergence(C)

        return divergence, P


class DifferentiableSinkhorn(nn.Module):
    """
    Memory-efficient differentiable Sinkhorn for large batches.

    Uses log-domain stabilization to avoid numerical issues.
    """
    def __init__(self, epsilon=0.1, n_iters=10):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def forward(self, C, a=None, b=None):
        """
        Log-domain Sinkhorn algorithm.

        Args:
            C: Cost matrix [batch_size, batch_size]
            a: Source marginals [batch_size] (log domain)
            b: Target marginals [batch_size] (log domain)

        Returns:
            log_P: Log of transport plan [batch_size, batch_size]
        """
        batch_size = C.size(0)

        # Default to uniform marginals in log domain
        if a is None:
            a = -torch.log(torch.tensor(batch_size, dtype=torch.float32))
        if b is None:
            b = -torch.log(torch.tensor(batch_size, dtype=torch.float32))

        # Log of kernel matrix
        log_K = -C / self.epsilon

        # Initialize in log domain
        f = torch.zeros(batch_size, device=C.device) + a
        g = torch.zeros(batch_size, device=C.device)

        # Sinkhorn iterations in log domain
        for _ in range(self.n_iters):
            # Update f
            log_sum_exp_g = torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
            f = a - log_sum_exp_g

            # Update g
            log_sum_exp_f = torch.logsumexp(log_K + f.unsqueeze(1), dim=0)
            g = b - log_sum_exp_f

        # Log transport plan
        log_P = f.unsqueeze(1) + log_K + g.unsqueeze(0)

        return log_P
