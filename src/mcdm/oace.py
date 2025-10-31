"""
Método OACE (Optimized Aggregation by Comprehensive Evaluation)
=============================================================

Implementação do método OACE para tomada de decisão multicritério.
Funcionalidades:
- Agregação ponderada separada para assertividade e custo
- Parâmetro lambda (λ) para balanceamento entre assertividade e custo
- Normalização min-max para cada grupo de critérios
- Suporte a critérios de benefício e custo
- Geração de rankings explicáveis

O OACE calcula:
- a_m: Agregação ponderada das métricas de assertividade (benefícios)
- c_m: Agregação ponderada das métricas de custo (normalizadas e invertidas)
- S_φ(m) = λ * a_m + (1-λ) * c_m

Utilizado como função de fitness no NAS para avaliar arquiteturas
considerando múltiplos critérios simultaneamente (accuracy, eficiência, complexidade).
"""

import numpy as np
from typing import List, Dict, Any, Optional
from utils import (
    normalize_matrix,
    calculate_weights,
    rank_alternatives
)


class OACE:
    """
    Implementação do método OACE (Optimized Aggregation by Comprehensive Evaluation).
    
    O OACE é um método MCDM que:
    1. Separa critérios em assertividade (benefícios) e custo
    2. Normaliza cada grupo separadamente usando min-max
    3. Calcula agregação ponderada para cada grupo
    4. Combina usando parâmetro lambda: S_φ(m) = λ * a_m + (1-λ) * c_m
    
    Características:
    - Permite balanceamento explícito entre assertividade e custo
    - Normalização dinâmica baseada em min/max do conjunto de dados
    - Adequado para problemas com critérios claramente separáveis em benefícios e custos
    """
    
    def __init__(self, 
                 criteria_weights: Optional[List[float]] = None,
                 criteria_types: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o método OACE.
        
        Args:
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios ("benefit" ou "cost")
            config: Configurações adicionais:
                - lambda_param: Parâmetro lambda (0-1) para balanceamento (padrão: 0.5)
                - normalization_method: Método de normalização (padrão: "min_max")
                - weight_method: Método de cálculo de pesos (padrão: "ahp")
        """
        self.criteria_weights = criteria_weights
        self.criteria_types = criteria_types or ["benefit"] * len(criteria_weights) if criteria_weights else []
        self.config = config or {}
        
        # Configurações OACE
        self.lambda_param = self.config.get("lambda_param", 0.5)
        self.normalization_method = self.config.get("normalization_method", "min_max")
        self.weight_method = self.config.get("weight_method", "ahp")
        
        # Validar lambda
        if not 0.0 <= self.lambda_param <= 1.0:
            raise ValueError("O parâmetro lambda_param deve estar no intervalo [0, 1]")
    
    def evaluate(self,
                 decision_matrix: np.ndarray,
                 criteria_weights: Optional[List[float]] = None,
                 criteria_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Avalia alternativas usando o método OACE.
        
        Args:
            decision_matrix: Matriz de decisão (alternativas x critérios)
            criteria_weights: Pesos dos critérios (opcional, sobrescreve os configurados)
            criteria_types: Tipos dos critérios (opcional, sobrescreve os configurados)
            
        Returns:
            Dicionário com resultados da avaliação contendo:
                - scores: Array com scores OACE para cada alternativa (maior = melhor)
                - assertiveness_scores: Array com agregação de assertividade (a_m)
                - cost_scores: Array com agregação de custo normalizada (c_m)
                - ranks: Array com rankings (1 = melhor)
                - sorted_indices: Índices ordenados por score (melhor primeiro)
                - lambda_param: Valor do parâmetro lambda usado
        """
        # Usar pesos e tipos fornecidos ou os configurados
        weights = criteria_weights or self.criteria_weights
        types = criteria_types or self.criteria_types
        
        if weights is None:
            raise ValueError("Pesos dos critérios devem ser fornecidos")
        
        if len(types) != decision_matrix.shape[1]:
            types = ["benefit"] * decision_matrix.shape[1]
        
        # Aplicar método OACE
        results = self._apply_oace(decision_matrix, weights, types)
        
        return results
    
    def _apply_oace(self,
                    decision_matrix: np.ndarray,
                    weights: List[float],
                    criteria_types: List[str]) -> Dict[str, Any]:
        """
        Aplica o algoritmo OACE.
        
        Args:
            decision_matrix: Matriz de decisão
            weights: Pesos dos critérios
            criteria_types: Tipos dos critérios
            
        Returns:
            Resultados do OACE
        """
        num_alternatives, num_criteria = decision_matrix.shape
        
        # Separar critérios em assertividade (benefícios) e custo
        assertiveness_indices = [i for i, t in enumerate(criteria_types) if t == "benefit"]
        cost_indices = [i for i, t in enumerate(criteria_types) if t == "cost"]
        
        if len(assertiveness_indices) == 0:
            raise ValueError("Deve haver pelo menos um critério de benefício (assertividade)")
        if len(cost_indices) == 0:
            raise ValueError("Deve haver pelo menos um critério de custo")
        
        # Extrair sub-matrizes
        assertiveness_matrix = decision_matrix[:, assertiveness_indices]
        cost_matrix = decision_matrix[:, cost_indices]
        
        # Extrair pesos
        assertiveness_weights = [weights[i] for i in assertiveness_indices]
        cost_weights = [weights[i] for i in cost_indices]
        
        # Normalizar pesos
        assertiveness_weights_array = calculate_weights(assertiveness_weights, method=self.weight_method)
        cost_weights_array = calculate_weights(cost_weights, method=self.weight_method)
        
        # Normalizar cada grupo separadamente
        # Para assertividade: maior é melhor (normalização min-max direta)
        normalized_assertiveness = self._normalize_group(
            assertiveness_matrix, 
            is_benefit=True
        )
        
        # Para custo: menor é melhor (normalização min-max invertida)
        normalized_cost = self._normalize_group(
            cost_matrix,
            is_benefit=False
        )
        
        # Calcular agregação ponderada para assertividade
        # a_m = Σ (w_j * norm_s_j)
        assertiveness_scores = np.sum(
            normalized_assertiveness * assertiveness_weights_array,
            axis=1
        )
        
        # Calcular agregação ponderada para custo (já normalizado e invertido)
        # c_m = Σ (w_j * (1 - norm_c_j))
        # Como normalized_cost já está invertido (maior = melhor), usamos diretamente
        cost_scores = np.sum(
            normalized_cost * cost_weights_array,
            axis=1
        )
        
        # Calcular score final OACE: S_φ(m) = λ * a_m + (1-λ) * c_m
        oace_scores = (self.lambda_param * assertiveness_scores) + \
                     ((1 - self.lambda_param) * cost_scores)
        
        # Rankear alternativas (maior score = melhor)
        ranks, sorted_indices = rank_alternatives(oace_scores, ascending=False)
        
        return {
            "scores": oace_scores,
            "assertiveness_scores": assertiveness_scores,
            "cost_scores": cost_scores,
            "ranks": ranks,
            "sorted_indices": sorted_indices,
            "lambda_param": self.lambda_param,
            "normalized_assertiveness": normalized_assertiveness,
            "normalized_cost": normalized_cost,
            "assertiveness_weights": assertiveness_weights_array,
            "cost_weights": cost_weights_array
        }
    
    def _normalize_group(self,
                        matrix: np.ndarray,
                        is_benefit: bool = True) -> np.ndarray:
        """
        Normaliza um grupo de critérios usando min-max.
        
        Args:
            matrix: Matriz do grupo (alternativas x critérios do grupo)
            is_benefit: Se True, critério de benefício (maior = melhor)
                       Se False, critério de custo (menor = melhor)
        
        Returns:
            Matriz normalizada entre 0 e 1
        """
        normalized = matrix.copy().astype(float)
        
        for j in range(matrix.shape[1]):
            column = matrix[:, j]
            min_val = np.min(column)
            max_val = np.max(column)
            
            if max_val != min_val:
                if is_benefit:
                    # Benefício: normalização direta (maior = melhor)
                    normalized[:, j] = (column - min_val) / (max_val - min_val)
                else:
                    # Custo: normalização invertida (menor = melhor)
                    # Maior valor fica menor normalizado, menor valor fica maior normalizado
                    normalized[:, j] = (max_val - column) / (max_val - min_val)
            else:
                # Se todos os valores são iguais, normalizar para 1.0
                normalized[:, j] = 1.0
        
        return normalized


# Teste básico
if __name__ == "__main__":
    import sys
    import os
    
    # Ajusta imports para execução direta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Dados de teste: modelos ML com métricas
    decision_matrix = np.array([
        [0.85, 0.86, 0.84, 0.83, 0.835, 5.2e6, 0.008, 220.0, 2.4],   # Modelo A
        [0.84, 0.88, 0.86, 0.85, 0.855, 7.8e6, 0.010, 300.0, 3.1],   # Modelo B
        [0.80, 0.84, 0.82, 0.81, 0.815, 3.5e6, 0.006, 180.0, 1.8],   # Modelo C
        [0.84, 0.87, 0.85, 0.84, 0.845, 6.1e6, 0.007, 210.0, 2.2],   # Modelo D
    ], dtype=float)
    
    # Pesos: 5 benefícios (accuracy, top5, precision, recall, f1) + 4 custos (params, time, mem, gflops)
    criteria_weights = [0.40, 0.15, 0.25, 0.15, 0.05, 0.25, 0.25, 0.25, 0.25]
    criteria_types = ["benefit"] * 5 + ["cost"] * 4
    
    print("="*70)
    print("TESTE OACE - Ranking de Modelos de ML")
    print("="*70)
    
    # Teste com diferentes valores de lambda
    for lambda_val in [0.0, 0.3, 0.5, 0.7, 1.0]:
        print(f"\n{'='*70}")
        print(f"Teste com lambda = {lambda_val}")
        print(f"{'='*70}")
        
        oace = OACE(
            criteria_weights=criteria_weights,
            criteria_types=criteria_types,
            config={"lambda_param": lambda_val, "normalization_method": "min_max"}
        )
        
        result = oace.evaluate(decision_matrix, criteria_weights, criteria_types)
        
        print(f"\nRanking:")
        sorted_idx = result["sorted_indices"]
        for pos, idx in enumerate(sorted_idx, start=1):
            modelo = ["A", "B", "C", "D"][idx]
            print(f"  {pos}º lugar: Modelo {modelo:8s} | "
                  f"Score OACE: {result['scores'][idx]:.6f} | "
                  f"Assertividade: {result['assertiveness_scores'][idx]:.6f} | "
                  f"Custo: {result['cost_scores'][idx]:.6f}")
        
        print(f"\nDetalhes:")
        print(f"  Lambda: {result['lambda_param']}")
        print(f"  Melhor: Modelo {['A', 'B', 'C', 'D'][sorted_idx[0]]} "
              f"(Score: {result['scores'][sorted_idx[0]]:.6f})")
    
    print("\n" + "="*70)
    print("Teste OACE concluído!")
    print("="*70)
