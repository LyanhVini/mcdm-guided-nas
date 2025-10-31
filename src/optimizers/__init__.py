"""
Módulo do Otimizador AFSA-PSO
============================

Implementação do algoritmo híbrido Artificial Fish Swarm Algorithm (AFSA) 
com Particle Swarm Optimization (PSO) para Neural Architecture Search.

Classes:
- AFSA_PSO_Optimizer: Otimizador híbrido principal
- Particle: Representação de uma partícula/solução
"""

from .afsa_pso import AFSA_PSO_Optimizer
from .afsa import *
from .pso import *

__all__ = ["AFSA_PSO_Optimizer", "Particle", "PSO"]
