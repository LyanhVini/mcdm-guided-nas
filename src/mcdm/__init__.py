"""
Módulo de Métodos MCDM
=====================

Este módulo contém implementações dos métodos de Tomada de Decisão Multicritério
usados como função de fitness no Neural Architecture Search.

Métodos implementados:
- OACE: Optimized Aggregation by Comprehensive Evaluation
- TOPSIS: Technique for Order Preference by Similarity to Ideal Solution
- VIKOR: VlseKriterijumska Optimizacija I Kompromisno Resenje

Utilitários:
- utils: Funções auxiliares para normalização e processamento
"""

from .oace import OACE
from .topsis import TOPSIS
from .vikor import VIKOR
from .utils import *

__all__ = ["OACE", "TOPSIS", "VIKOR"]
