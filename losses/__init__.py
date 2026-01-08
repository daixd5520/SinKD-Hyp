"""
MoG-SKD: Mixture-of-Geometries Sinkhorn Knowledge Distillation

A unified framework combining multiple geometric experts for knowledge distillation.
"""

from .experts import (
    GeometryExpert,
    FisherRaoExpert,
    EuclideanExpert,
    HyperbolicExpert
)
from .gating import StatisticalGating
from .sinkhorn import SinkhornSolver

__all__ = [
    'GeometryExpert',
    'FisherRaoExpert',
    'EuclideanExpert',
    'HyperbolicExpert',
    'StatisticalGating',
    'SinkhornSolver'
]
