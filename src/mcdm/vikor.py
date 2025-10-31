"""
Método VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)
================================================================

Implementação do método VIKOR para tomada de decisão multicritério.
Funcionalidades:
- Método de compromisso entre benefício máximo e arrependimento mínimo
- Cálculo de valores de utilidade (S) e arrependimento (R)
- Índice de compromisso (Q) para ranking de alternativas
- Verificação de condições de parada para soluções únicas/múltiplas
- Análise de sensibilidade para robustez das decisões

Utilizado como função de fitness no NAS para encontrar arquiteturas
que oferecem o melhor equilíbrio entre diferentes critérios de performance.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from utils import (
    normalize_matrix, 
    calculate_weights, 
    rank_alternatives
)


class VIKOR:
    """
    Implementação do método VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje).
    
    O VIKOR é um método MCDM que:
    1. Normaliza a matriz de decisão
    2. Calcula valores de utilidade e arrependimento
    3. Calcula índice de compromisso
    4. Identifica soluções de compromisso
    5. Rankeia alternativas baseado no índice
    """
    
    def __init__(self, 
                 criteria_weights: Optional[List[float]] = None,
                 criteria_types: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o método VIKOR.
        
        Args:
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios ("benefit" ou "cost")
            config: Configurações adicionais
        """
        self.criteria_weights = criteria_weights
        self.criteria_types = criteria_types or ["benefit"] * len(criteria_weights) if criteria_weights else []
        self.config = config or {}
        
        # Parâmetro de estratégia de grupo (v)
        self.v = self.config.get("v", 0.5)  # 0.5 = estratégia de compromisso
        
        # Configurações de normalização
        self.normalization_method = self.config.get("normalization_method", "min_max")
        self.weight_method = self.config.get("weight_method", "ahp")
        
        # Configurações de condições de parada
        self.condition1_threshold = self.config.get("condition1_threshold", 0.25)
        self.condition2_threshold = self.config.get("condition2_threshold", 0.1)
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, 
                decision_matrix: np.ndarray,
                criteria_weights: Optional[List[float]] = None,
                criteria_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Avalia alternativas usando o método VIKOR.
        
        Args:
            decision_matrix: Matriz de decisão (alternativas x critérios)
            criteria_weights: Pesos dos critérios (opcional)
            criteria_types: Tipos dos critérios (opcional)
            
        Returns:
            Dicionário com resultados da avaliação
        """
        # Usar pesos e tipos fornecidos ou os configurados
        weights = criteria_weights or self.criteria_weights
        types = criteria_types or self.criteria_types
        
        if weights is None:
            raise ValueError("Pesos dos critérios devem ser fornecidos")
        
        if len(types) != decision_matrix.shape[1]:
            types = ["benefit"] * decision_matrix.shape[1]
        
        # Aplicar método VIKOR
        results = self._apply_vikor(decision_matrix, weights, types)
        
        return results
    
    def _apply_vikor(self, 
                    decision_matrix: np.ndarray,
                    weights: List[float],
                    criteria_types: List[str]) -> Dict[str, Any]:
        """
        Aplica o algoritmo VIKOR.
        
        Args:
            decision_matrix: Matriz de decisão
            weights: Pesos dos critérios
            criteria_types: Tipos dos critérios
            
        Returns:
            Resultados do VIKOR
        """
        num_alternatives, num_criteria = decision_matrix.shape
        
        # Etapa 1: Normalizar matriz de decisão
        normalized_matrix = normalize_matrix(
            decision_matrix, 
            method=self.normalization_method,
            criteria_types=criteria_types
        )
        
        # Etapa 2: Calcular pesos normalizados
        weights_array = calculate_weights(weights, method=self.weight_method)
        
        # Etapa 3: Identificar valores ideais e anti-ideais
        ideal_values, anti_ideal_values = self._find_ideal_anti_ideal_values(
            normalized_matrix, criteria_types
        )
        
        # Etapa 4: Calcular valores de utilidade (S) e arrependimento (R)
        utility_values = self._calculate_utility_values(
            normalized_matrix, weights_array, ideal_values, anti_ideal_values
        )
        
        regret_values = self._calculate_regret_values(
            normalized_matrix, weights_array, ideal_values, anti_ideal_values
        )
        
        # Etapa 5: Calcular índice de compromisso (Q)
        compromise_index = self._calculate_compromise_index(
            utility_values, regret_values
        )
        
        # Etapa 6: Rankear alternativas
        q_ranks, q_sorted_indices = rank_alternatives(compromise_index, ascending=True)
        s_ranks, s_sorted_indices = rank_alternatives(utility_values, ascending=True)
        r_ranks, r_sorted_indices = rank_alternatives(regret_values, ascending=True)
        
        # Etapa 7: Verificar condições de parada
        conditions = self._check_conditions(compromise_index, utility_values, regret_values, q_ranks, s_ranks, r_ranks)
        
        # Calcular métricas adicionais
        metrics = self._calculate_vikor_metrics(
            decision_matrix,
            normalized_matrix,
            utility_values,
            regret_values,
            compromise_index,
            ideal_values,
            anti_ideal_values
        )
        
        return {
            "scores": compromise_index,  # Q values
            "ranks": q_ranks,
            "sorted_indices": q_sorted_indices,
            "utility_values": utility_values,  # S values
            "regret_values": regret_values,    # R values
            "utility_ranks": s_ranks,
            "regret_ranks": r_ranks,
            "ideal_values": ideal_values,
            "anti_ideal_values": anti_ideal_values,
            "normalized_matrix": normalized_matrix,
            "weights": weights_array,
            "conditions": conditions,
            "metrics": metrics
        }
    
    def _find_ideal_anti_ideal_values(self, 
                                    normalized_matrix: np.ndarray,
                                    criteria_types: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encontra valores ideais e anti-ideais para cada critério.
        
        Args:
            normalized_matrix: Matriz normalizada
            criteria_types: Tipos dos critérios
            
        Returns:
            Tupla com (valores_ideais, valores_anti_ideais)
        """
        num_criteria = normalized_matrix.shape[1]
        ideal_values = np.zeros(num_criteria)
        anti_ideal_values = np.zeros(num_criteria)
        
        for j in range(num_criteria):
            criterion_type = criteria_types[j]
            column = normalized_matrix[:, j]
            
            if criterion_type == "benefit":
                ideal_values[j] = np.max(column)
                anti_ideal_values[j] = np.min(column)
            else:  # cost
                ideal_values[j] = np.min(column)
                anti_ideal_values[j] = np.max(column)
        
        return ideal_values, anti_ideal_values
    
    def _calculate_utility_values(self, 
                                normalized_matrix: np.ndarray,
                                weights: np.ndarray,
                                ideal_values: np.ndarray,
                                anti_ideal_values: np.ndarray) -> np.ndarray:
        """
        Calcula valores de utilidade (S) para cada alternativa.
        
        Args:
            normalized_matrix: Matriz normalizada
            weights: Pesos dos critérios
            ideal_values: Valores ideais
            anti_ideal_values: Valores anti-ideais
            
        Returns:
            Array com valores de utilidade
        """
        num_alternatives = normalized_matrix.shape[0]
        utility_values = np.zeros(num_alternatives)
        
        for i in range(num_alternatives):
            utility_sum = 0.0
            for j in range(len(weights)):
                if ideal_values[j] != anti_ideal_values[j]:
                    utility_sum += weights[j] * (ideal_values[j] - normalized_matrix[i, j]) / (ideal_values[j] - anti_ideal_values[j])
            utility_values[i] = utility_sum
        
        return utility_values
    
    def _calculate_regret_values(self, 
                               normalized_matrix: np.ndarray,
                               weights: np.ndarray,
                               ideal_values: np.ndarray,
                               anti_ideal_values: np.ndarray) -> np.ndarray:
        """
        Calcula valores de arrependimento (R) para cada alternativa.
        
        Args:
            normalized_matrix: Matriz normalizada
            weights: Pesos dos critérios
            ideal_values: Valores ideais
            anti_ideal_values: Valores anti-ideais
            
        Returns:
            Array com valores de arrependimento
        """
        num_alternatives = normalized_matrix.shape[0]
        regret_values = np.zeros(num_alternatives)
        
        for i in range(num_alternatives):
            max_regret = 0.0
            for j in range(len(weights)):
                if ideal_values[j] != anti_ideal_values[j]:
                    regret = weights[j] * (ideal_values[j] - normalized_matrix[i, j]) / (ideal_values[j] - anti_ideal_values[j])
                    max_regret = max(max_regret, regret)
            regret_values[i] = max_regret
        
        return regret_values
    
    def _calculate_compromise_index(self, 
                                  utility_values: np.ndarray,
                                  regret_values: np.ndarray) -> np.ndarray:
        """
        Calcula índice de compromisso (Q) para cada alternativa.
        
        Args:
            utility_values: Valores de utilidade (S)
            regret_values: Valores de arrependimento (R)
            
        Returns:
            Array com índices de compromisso
        """
        # Encontrar valores mínimos e máximos
        s_min = np.min(utility_values)
        s_max = np.max(utility_values)
        r_min = np.min(regret_values)
        r_max = np.max(regret_values)
        
        # Calcular índices de compromisso
        compromise_index = np.zeros(len(utility_values))
        
        for i in range(len(utility_values)):
            if s_max != s_min and r_max != r_min:
                s_component = (utility_values[i] - s_min) / (s_max - s_min)
                r_component = (regret_values[i] - r_min) / (r_max - r_min)
                compromise_index[i] = self.v * s_component + (1 - self.v) * r_component
            else:
                compromise_index[i] = 0.0
        
        return compromise_index
    
    def _check_conditions(self, 
                         compromise_index: np.ndarray,
                         utility_values: np.ndarray,
                         regret_values: np.ndarray,
                         q_ranks: np.ndarray,
                         s_ranks: np.ndarray,
                         r_ranks: np.ndarray) -> Dict[str, Any]:
        """
        Verifica condições de parada do VIKOR.
        
        Args:
            compromise_index: Valores Q (índice de compromisso)
            utility_values: Valores S (utilidade)
            regret_values: Valores R (arrependimento)
            q_ranks: Rankings do índice de compromisso
            s_ranks: Rankings dos valores de utilidade
            r_ranks: Rankings dos valores de arrependimento
            
        Returns:
            Dicionário com status das condições
        """
        # Condição 1: Aceitável vantagem (usa valores Q, não ranks!)
        best_alternative = np.argmin(compromise_index)
        sorted_indices = np.argsort(compromise_index)
        second_best_alternative = sorted_indices[1] if len(sorted_indices) > 1 else best_alternative
        
        q_best = compromise_index[best_alternative]
        q_second = compromise_index[second_best_alternative]
        
        threshold = 1.0 / (len(compromise_index) - 1)
        condition1 = (q_second - q_best) >= threshold
        
        # Condição 2: Aceitável estabilidade
        condition2 = (s_ranks[best_alternative] == 0) or (r_ranks[best_alternative] == 0)
        
        # Determinar se a solução é única ou se há múltiplas soluções
        if condition1 and condition2:
            solution_type = "unique"
        elif condition1 and not condition2:
            solution_type = "multiple_s_rank"
        elif not condition1 and condition2:
            solution_type = "multiple_r_rank"
        else:
            solution_type = "multiple_both"
        
        return {
            "condition1_acceptable_advantage": condition1,
            "condition2_acceptable_stability": condition2,
            "solution_type": solution_type,
            "best_alternative": best_alternative,
            "q_difference": q_second - q_best,
            "threshold": threshold
        }
    
    def _calculate_vikor_metrics(self, 
                               decision_matrix: np.ndarray,
                               normalized_matrix: np.ndarray,
                               utility_values: np.ndarray,
                               regret_values: np.ndarray,
                               compromise_index: np.ndarray,
                               ideal_values: np.ndarray,
                               anti_ideal_values: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas adicionais do VIKOR.
        
        Args:
            decision_matrix: Matriz de decisão original
            normalized_matrix: Matriz normalizada
            utility_values: Valores de utilidade
            regret_values: Valores de arrependimento
            compromise_index: Índices de compromisso
            ideal_values: Valores ideais
            anti_ideal_values: Valores anti-ideais
            
        Returns:
            Dicionário com métricas
        """
        # Variância dos índices de compromisso
        q_variance = np.var(compromise_index)
        
        # Diferença entre melhor e pior alternativa
        q_difference = np.max(compromise_index) - np.min(compromise_index)
        
        # Correlação entre S e R
        s_r_correlation = np.corrcoef(utility_values, regret_values)[0, 1]
        
        # Entropia dos índices de compromisso
        q_entropy = -np.sum(compromise_index * np.log(compromise_index + 1e-10))
        
        # Coeficiente de variação
        q_cv = np.std(compromise_index) / (np.mean(compromise_index) + 1e-10)
        
        # Distância média dos valores ideais
        avg_distance_ideal = np.mean([np.linalg.norm(normalized_matrix[i, :] - ideal_values) 
                                    for i in range(len(utility_values))])
        
        return {
            "q_variance": q_variance,
            "q_difference": q_difference,
            "s_r_correlation": s_r_correlation,
            "q_entropy": q_entropy,
            "q_coefficient_of_variation": q_cv,
            "avg_distance_ideal": avg_distance_ideal,
            "num_alternatives": len(compromise_index),
            "num_criteria": len(ideal_values)
        }
    
    def get_compromise_solutions(self, 
                               decision_matrix: np.ndarray,
                               criteria_weights: List[float],
                               criteria_types: List[str]) -> Dict[str, Any]:
        """
        Identifica soluções de compromisso baseadas nas condições VIKOR.
        
        Args:
            decision_matrix: Matriz de decisão
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios
            
        Returns:
            Soluções de compromisso identificadas
        """
        result = self.evaluate(decision_matrix, criteria_weights, criteria_types)
        conditions = result["conditions"]
        
        compromise_solutions = []
        
        if conditions["solution_type"] == "unique":
            # Solução única
            best_alt = conditions["best_alternative"]
            compromise_solutions.append({
                "alternative": best_alt,
                "type": "unique",
                "q_value": result["scores"][best_alt],
                "s_value": result["utility_values"][best_alt],
                "r_value": result["regret_values"][best_alt]
            })
        
        else:
            # Múltiplas soluções
            q_sorted_indices = result["sorted_indices"]
            
            # Adicionar melhor alternativa
            best_alt = q_sorted_indices[0]
            compromise_solutions.append({
                "alternative": best_alt,
                "type": "best",
                "q_value": result["scores"][best_alt],
                "s_value": result["utility_values"][best_alt],
                "r_value": result["regret_values"][best_alt]
            })
            
            # Adicionar alternativas adicionais se necessário
            if conditions["solution_type"] in ["multiple_s_rank", "multiple_both"]:
                # Encontrar alternativa com melhor S
                best_s_alt = np.argmin(result["utility_values"])
                if best_s_alt != best_alt:
                    compromise_solutions.append({
                        "alternative": best_s_alt,
                        "type": "best_s",
                        "q_value": result["scores"][best_s_alt],
                        "s_value": result["utility_values"][best_s_alt],
                        "r_value": result["regret_values"][best_s_alt]
                    })
            
            if conditions["solution_type"] in ["multiple_r_rank", "multiple_both"]:
                # Encontrar alternativa com melhor R
                best_r_alt = np.argmin(result["regret_values"])
                if best_r_alt != best_alt and best_r_alt not in [sol["alternative"] for sol in compromise_solutions]:
                    compromise_solutions.append({
                        "alternative": best_r_alt,
                        "type": "best_r",
                        "q_value": result["scores"][best_r_alt],
                        "s_value": result["utility_values"][best_r_alt],
                        "r_value": result["regret_values"][best_r_alt]
                    })
        
        return {
            "solution_type": conditions["solution_type"],
            "compromise_solutions": compromise_solutions,
            "conditions": conditions
        }
    
    def calculate_sensitivity_analysis(self, 
                                     decision_matrix: np.ndarray,
                                     criteria_weights: List[float],
                                     criteria_types: List[str],
                                     weight_changes: List[float] = None) -> Dict[str, Any]:
        """
        Realiza análise de sensibilidade do método VIKOR.
        
        Args:
            decision_matrix: Matriz de decisão
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios
            weight_changes: Lista de mudanças percentuais nos pesos
            
        Returns:
            Resultados da análise de sensibilidade
        """
        if weight_changes is None:
            weight_changes = [-0.2, -0.1, 0.1, 0.2]
        
        # Resultado original
        original_result = self.evaluate(decision_matrix, criteria_weights, criteria_types)
        original_ranking = original_result["ranks"]
        original_q_scores = original_result["scores"]
        
        sensitivity_results = {}
        
        for i, weight in enumerate(criteria_weights):
            criterion_sensitivity = {}
            
            for change in weight_changes:
                # Modificar peso do critério i
                modified_weights = criteria_weights.copy()
                modified_weights[i] = weight * (1 + change)
                
                # Renormalizar pesos
                modified_weights = np.array(modified_weights)
                modified_weights = modified_weights / np.sum(modified_weights)
                
                # Avaliar com pesos modificados
                modified_result = self.evaluate(decision_matrix, modified_weights.tolist(), criteria_types)
                modified_ranking = modified_result["ranks"]
                modified_q_scores = modified_result["scores"]
                
                # Calcular mudanças
                ranking_change = np.sum(np.abs(modified_ranking - original_ranking))
                q_score_change = np.mean(np.abs(modified_q_scores - original_q_scores))
                
                criterion_sensitivity[f"change_{change:.1f}"] = {
                    "ranking_change": ranking_change,
                    "q_score_change": q_score_change,
                    "modified_ranking": modified_ranking.tolist(),
                    "modified_q_scores": modified_q_scores.tolist(),
                    "modified_weights": modified_weights.tolist(),
                    "conditions": modified_result["conditions"]
                }
            
            sensitivity_results[f"criterion_{i}"] = criterion_sensitivity
        
        return {
            "original_ranking": original_ranking.tolist(),
            "original_q_scores": original_q_scores.tolist(),
            "original_conditions": original_result["conditions"],
            "sensitivity_results": sensitivity_results
        }

if __name__ == "__main__":
    # Teste simples do VIKOR com uma matriz simulada de métricas de ML
    # Alternativas (linhas): Modelos A, B, C, D
    # Critérios (colunas): [top1_acc, precision_macro, recall_macro, f1_macro, total_params, avg_inference_time, memory_used_mb, gflops]
    import numpy as np

    decision_matrix = np.array([
        [0.85, 0.86, 0.84, 0.83, 0.835, 5.2e6, 0.008, 220.0, 2.4],   # Modelo A
        [0.84, 0.88, 0.86, 0.85, 0.855, 7.8e6, 0.010, 300.0, 3.1],   # Modelo B
        [0.80, 0.84, 0.82, 0.81, 0.815, 3.5e6, 0.006, 180.0, 1.8],   # Modelo C
        [0.84, 0.87, 0.85, 0.84, 0.845, 6.1e6, 0.007, 210.0, 2.2],   # Modelo D
    ], dtype=float)

    # Pesos e tipos
    criteria_weights = [
        0.40,  # top1_acc (benefit)
        0.15,  # top5_acc (benefit)
        0.25,  # precision_macro (benefit)
        0.15,  # recall_macro (benefit)
        0.05,  # f1_macro (benefit)
        0.25,  # total_params (cost)
        0.25,  # avg_inference_time (cost)
        0.25,  # memory_used_mb (cost)
        0.25,  # gflops (cost)
    ]

    criteria_types = [
        "benefit", "benefit", "benefit", "benefit", "benefit",
        "cost", "cost", "cost", "cost",
    ]

    # Instancia VIKOR e avalia
    vikor = VIKOR(
        criteria_weights=criteria_weights,
        criteria_types=criteria_types,
        config={
            "v": 0.7,
            "normalization_method": "vector",  # Alterado de "vector" para "min_max" - mais adequado para VIKOR
            "weight_method": "ahp",
            "condition1_threshold": 0.25,
            "condition2_threshold": 0.1,
        }
    )

    result = vikor.evaluate(
        decision_matrix=decision_matrix,
        criteria_weights=criteria_weights,
        criteria_types=criteria_types,
    )

    # Impressão dos resultados
    print("\n" + "="*60)
    print("TESTE VIKOR - Ranking de Modelos de ML")
    print("="*60)

    model_names = ["Modelo A", "Modelo B", "Modelo C", "Modelo D"]
    q_scores = result["scores"]
    s_values = result["utility_values"]
    r_values = result["regret_values"]
    ranks = result["ranks"]
    sorted_idx = result["sorted_indices"]

    print("\n📊 RANKING (por Q - índice de compromisso):")
    print("-" * 60)
    for pos, idx in enumerate(sorted_idx, start=1):
        print(f"{pos}º lugar: {model_names[idx]:12s} | Q: {q_scores[idx]:.6f} | S: {s_values[idx]:.6f} | R: {r_values[idx]:.6f} | Rank: {ranks[idx] + 1}")

    print("\n📋 CONDIÇÕES VIKOR:")
    print("-" * 60)
    conditions = result["conditions"]
    print(f"Tipo de solução: {conditions['solution_type']}")
    print(f"Vantagem aceitável (Cond.1): {conditions['condition1_acceptable_advantage']}")
    print(f"Estabilidade aceitável (Cond.2): {conditions['condition2_acceptable_stability']}")
    print(f"Diferença Q (segundo - melhor): {conditions['q_difference']:.6f} | Limite: {conditions['threshold']:.6f}")
    print(f"Melhor alternativa (índice): {conditions['best_alternative']} ({model_names[conditions['best_alternative']]})")
    
    # Validação da Condição 1
    best_idx = conditions['best_alternative']
    q_best_val = q_scores[best_idx]
    second_best_idx = np.argsort(q_scores)[1]
    q_second_val = q_scores[second_best_idx]
    print(f"\nValidação Cond.1: Q({model_names[second_best_idx]}) - Q({model_names[best_idx]}) = {q_second_val:.6f} - {q_best_val:.6f} = {conditions['q_difference']:.6f}")
    print(f"  {conditions['q_difference']:.6f} >= {conditions['threshold']:.6f} ? {conditions['condition1_acceptable_advantage']}")

    print("\n🎯 VALORES IDEAIS (normalizados):")
    print("-" * 60)
    print("Ideais:      ", result["ideal_values"])
    print("Anti-ideais: ", result["anti_ideal_values"])
    
    print("\n📊 DETALHES POR MODELO:")
    print("-" * 60)
    print(f"{'Modelo':<12} {'Q (Score)':>12} {'S (Util)':>12} {'R (Reg)':>12} {'Rank Q':>8} {'Rank S':>8} {'Rank R':>8}")
    for i, name in enumerate(model_names):
        print(f"{name:<12} {q_scores[i]:>12.6f} {s_values[i]:>12.6f} {r_values[i]:>12.6f} "
              f"{ranks[i]+1:>8} {result['utility_ranks'][i]+1:>8} {result['regret_ranks'][i]+1:>8}")

    print("\n📈 ANÁLISE:")
    print("-" * 60)
    print(f"Modelo com menor Q (melhor compromisso): {model_names[best_idx]} (Q={q_scores[best_idx]:.6f})")
    print(f"Modelo com menor S (melhor utilidade): {model_names[np.argmin(s_values)]} (S={np.min(s_values):.6f})")
    print(f"Modelo com menor R (menor arrependimento): {model_names[np.argmin(r_values)]} (R={np.min(r_values):.6f})")
    print(f"\nDiferença entre melhor e pior Q: {np.max(q_scores) - np.min(q_scores):.6f}")
    
    print("\n✅ Teste VIKOR concluído!")
    print("="*60)
