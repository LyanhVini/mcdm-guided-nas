"""
Wrapper para integração de MCDMs (OACE, TOPSIS, VIKOR) com PSO
===============================================================

Gerencia avaliação de partículas do PSO usando diferentes métodos MCDM.
Permite comparação justa entre métodos através de avaliação por geração.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
import sys
import os

# Ajusta path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcdm.oace import OACE
from mcdm.topsis import TOPSIS
from mcdm.vikor import VIKOR
from mcdm.utils import build_decision_matrix, mcdm_to_fitness


class PSOMCDMWrapper:
    """
    Wrapper que integra MCDMs com PSO para avaliação de partículas.
    
    Gerencia:
    - Treinamento de partículas
    - Construção de matriz de decisão
    - Aplicação do MCDM escolhido
    - Conversão de scores para fitness (PSO minimiza)
    """
    
    # Mapeamento de métodos para tipo de avaliação
    EVALUATION_TYPES = {
        "OACE": "individual",      # Avalia uma partícula por vez
        "TOPSIS": "contextual",    # Precisa de todas as partículas
        "VIKOR": "contextual"      # Precisa de todas as partículas
    }
    
    def __init__(self,
                 mcdm_method: str,
                 train_function: Callable,
                 criteria_weights: List[float],
                 criteria_types: List[str],
                 mcdm_config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o wrapper.
        
        Args:
            mcdm_method: Método MCDM ("OACE", "TOPSIS", ou "VIKOR")
            train_function: Função que recebe posição (array) e retorna dict com métricas
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios ("benefit" ou "cost")
            mcdm_config: Configurações específicas do MCDM (lambda para OACE, v para VIKOR, etc.)
        """
        if mcdm_method not in self.EVALUATION_TYPES:
            raise ValueError(f"Método MCDM inválido: {mcdm_method}. Use: OACE, TOPSIS ou VIKOR")
        
        self.mcdm_method = mcdm_method
        self.evaluation_type = self.EVALUATION_TYPES[mcdm_method]
        self.train_function = train_function
        self.criteria_weights = criteria_weights
        self.criteria_types = criteria_types
        self.mcdm_config = mcdm_config or {}
        
        # Inicializa o avaliador MCDM
        self.mcdm_evaluator = self._create_mcdm_evaluator()
        
        # Para OACE: manter histórico de limites (min/max por critério)
        if mcdm_method == "OACE":
            self.metrics_history = {
                "assertiveness": {},  # min/max para cada métrica de assertividade
                "cost": {}           # min/max para cada métrica de custo
            }
    
    def _create_mcdm_evaluator(self):
        """Cria o avaliador MCDM apropriado."""
        config = self.mcdm_config.copy()
        
        if self.mcdm_method == "OACE":
            return OACE(
                criteria_weights=self.criteria_weights,
                criteria_types=self.criteria_types,
                config=config
            )
        elif self.mcdm_method == "TOPSIS":
            return TOPSIS(
                criteria_weights=self.criteria_weights,
                criteria_types=self.criteria_types,
                config=config
            )
        elif self.mcdm_method == "VIKOR":
            return VIKOR(
                criteria_weights=self.criteria_weights,
                criteria_types=self.criteria_types,
                config=config
            )
    
    def fitness_function(self, particles_positions: np.ndarray) -> np.ndarray:
        """
        Avalia um batch de partículas usando o MCDM configurado.
        
        Args:
            particles_positions: Array (n_particles, n_dim) com posições das partículas
        
        Returns:
            Array (n_particles,) com valores de fitness (menor = melhor para PSO)
        """
        n_particles = particles_positions.shape[0]
        
        # Fase 1: Treinar todas as partículas e coletar métricas
        all_metrics = []
        for i in range(n_particles):
            metrics = self.train_function(particles_positions[i])
            all_metrics.append(metrics)
        
        # Fase 2: Construir matriz de decisão
        decision_matrix = build_decision_matrix(all_metrics)
        
        # Fase 3: Avaliar usando MCDM apropriado
        if self.evaluation_type == "individual":
            scores = self._evaluate_individual(all_metrics, decision_matrix)
        else:  # contextual
            scores = self._evaluate_contextual(decision_matrix)
        
        # Fase 4: Converter scores para fitness (PSO minimiza)
        fitness = mcdm_to_fitness(scores, invert=True)
        
        return fitness
    
    
    def _evaluate_individual(self, all_metrics: List[Dict], decision_matrix: np.ndarray) -> np.ndarray:
        """Avalia cada partícula individualmente (OACE)."""
        scores = []
        
        for i, metrics in enumerate(all_metrics):
            # Atualiza limites históricos para OACE
            self._update_oace_limits(metrics)
            
            # Avalia individualmente
            row = decision_matrix[i:i+1, :]  # Shape (1, n_criteria)
            result = self.mcdm_evaluator.evaluate(
                row,
                self.criteria_weights,
                self.criteria_types
            )
            scores.append(result["scores"][0])
        
        return np.array(scores)
    
    def _evaluate_contextual(self, decision_matrix: np.ndarray) -> np.ndarray:
        """Avalia geração completa contextualmente (TOPSIS ou VIKOR)."""
        result = self.mcdm_evaluator.evaluate(
            decision_matrix,
            self.criteria_weights,
            self.criteria_types
        )
        return result["scores"]
    
    def _update_oace_limits(self, metrics: Dict[str, float]):
        """Atualiza limites históricos min/max para OACE."""
        if self.mcdm_method != "OACE":
            return
        
        assertiveness_keys = ["top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro"]
        cost_keys = ["total_params", "avg_inference_time", "memory_used_mb", "gflops"]
        
        # Atualiza assertividade
        for key in assertiveness_keys:
            if key in metrics:
                if key not in self.metrics_history["assertiveness"]:
                    self.metrics_history["assertiveness"][key] = {
                        "min": metrics[key],
                        "max": metrics[key]
                    }
                else:
                    self.metrics_history["assertiveness"][key]["min"] = min(
                        self.metrics_history["assertiveness"][key]["min"],
                        metrics[key]
                    )
                    self.metrics_history["assertiveness"][key]["max"] = max(
                        self.metrics_history["assertiveness"][key]["max"],
                        metrics[key]
                    )
        
        # Atualiza custo
        for key in cost_keys:
            if key in metrics:
                if key not in self.metrics_history["cost"]:
                    self.metrics_history["cost"][key] = {
                        "min": metrics[key],
                        "max": metrics[key]
                    }
                else:
                    self.metrics_history["cost"][key]["min"] = min(
                        self.metrics_history["cost"][key]["min"],
                        metrics[key]
                    )
                    self.metrics_history["cost"][key]["max"] = max(
                        self.metrics_history["cost"][key]["max"],
                        metrics[key]
                    )
    
    @staticmethod
    def is_contextual(mcdm_method: str) -> bool:
        """
        Verifica se um método MCDM requer avaliação contextual.
        
        Args:
            mcdm_method: Nome do método ("OACE", "TOPSIS", "VIKOR")
        
        Returns:
            True se requer contexto (todas as partículas), False caso contrário
        """
        return PSOMCDMWrapper.EVALUATION_TYPES.get(mcdm_method, "contextual") == "contextual"
    
    @staticmethod
    def is_individual(mcdm_method: str) -> bool:
        """
        Verifica se um método MCDM permite avaliação individual.
        
        Args:
            mcdm_method: Nome do método ("OACE", "TOPSIS", "VIKOR")
        
        Returns:
            True se permite avaliação individual, False caso contrário
        """
        return PSOMCDMWrapper.EVALUATION_TYPES.get(mcdm_method, "contextual") == "individual"

