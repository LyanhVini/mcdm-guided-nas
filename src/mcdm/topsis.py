"""
Método TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
============================================================================

Implementação do método TOPSIS para tomada de decisão multicritério.
Funcionalidades:
- Identificação da alternativa mais próxima da solução ideal positiva
- Cálculo de distâncias para soluções ideais positiva e negativa
- Coeficiente de proximidade relativa para ranking
- Análise de sensibilidade para robustez das decisões
- Suporte a diferentes métricas de distância e normalização

Utilizado como função de fitness no NAS para avaliar arquiteturas
baseado na proximidade com soluções ideais de performance.

BASE -> https://github.com/Glitchfix/TOPSIS-Python/tree/master
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from utils import (
    normalize_matrix, 
    calculate_weights, 
    calculate_distance,
    find_ideal_solutions,
    rank_alternatives
)

#
class TOPSIS:
    """
    Implementação do método TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
    
    O TOPSIS é um método MCDM que:
    1. Normaliza a matriz de decisão
    2. Identifica soluções ideais positiva e negativa
    3. Calcula distâncias para cada solução ideal
    4. Calcula coeficiente de proximidade relativa
    5. Rankeia alternativas baseado no coeficiente
    """
    
    def __init__(self, 
                 criteria_weights: Optional[List[float]] = None,
                 criteria_types: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o método TOPSIS.
        
        Args:
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios ("benefit" ou "cost")
            config: Configurações adicionais
        """
        self.criteria_weights = criteria_weights
        self.criteria_types = criteria_types or ["benefit"] * len(criteria_weights) if criteria_weights else []
        self.config = config or {}
        
        # Configurações TOPSIS
        self.normalization_method = self.config.get("normalization_method", "vector")
        self.distance_metric = self.config.get("distance_metric", "euclidean")
        self.weight_method = self.config.get("weight_method", "ahp")
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, 
                decision_matrix: np.ndarray,
                criteria_weights: Optional[List[float]] = None,
                criteria_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Avalia alternativas usando o método TOPSIS.
        
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
        
        # Aplicar método TOPSIS
        results = self._apply_topsis(decision_matrix, weights, types)
        
        return results
    
    def _apply_topsis(self, 
                     decision_matrix: np.ndarray,
                     weights: List[float],
                     criteria_types: List[str]) -> Dict[str, Any]:
        """
        Aplica o algoritmo TOPSIS.
        
        Args:
            decision_matrix: Matriz de decisão
            weights: Pesos dos critérios
            criteria_types: Tipos dos critérios
            
        Returns:
            Resultados do TOPSIS
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
        
        # Etapa 3: Aplicar pesos à matriz normalizada
        weighted_matrix = normalized_matrix * weights_array
        
        # Etapa 4: Identificar soluções ideais
        positive_ideal, negative_ideal = find_ideal_solutions(weighted_matrix, criteria_types)
        
        # Etapa 5: Calcular distâncias para soluções ideais
        distances_positive = np.zeros(num_alternatives)
        distances_negative = np.zeros(num_alternatives)
        
        for i in range(num_alternatives):
            distances_positive[i] = calculate_distance(
                weighted_matrix[i, :], 
                positive_ideal, 
                method=self.distance_metric
            )
            distances_negative[i] = calculate_distance(
                weighted_matrix[i, :], 
                negative_ideal, 
                method=self.distance_metric
            )
        
        # Etapa 6: Calcular coeficiente de proximidade relativa
        closeness_coefficients = distances_negative / (distances_positive + distances_negative)
        
        # Etapa 7: Rankear alternativas
        ranks, sorted_indices = rank_alternatives(closeness_coefficients, ascending=False)
        
        # Calcular métricas adicionais
        metrics = self._calculate_topsis_metrics(
            decision_matrix,
            normalized_matrix,
            weighted_matrix,
            positive_ideal,
            negative_ideal,
            distances_positive,
            distances_negative,
            closeness_coefficients
        )
        
        return {
            "scores": closeness_coefficients,
            "ranks": ranks,
            "sorted_indices": sorted_indices,
            "distances_positive": distances_positive,
            "distances_negative": distances_negative,
            "positive_ideal": positive_ideal,
            "negative_ideal": negative_ideal,
            "normalized_matrix": normalized_matrix,
            "weighted_matrix": weighted_matrix,
            "weights": weights_array,
            "metrics": metrics
        }
    
    def _calculate_topsis_metrics(self, 
                                 decision_matrix: np.ndarray,
                                 normalized_matrix: np.ndarray,
                                 weighted_matrix: np.ndarray,
                                 positive_ideal: np.ndarray,
                                 negative_ideal: np.ndarray,
                                 distances_positive: np.ndarray,
                                 distances_negative: np.ndarray,
                                 closeness_coefficients: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas adicionais do TOPSIS.
        
        Args:
            decision_matrix: Matriz de decisão original
            normalized_matrix: Matriz normalizada
            weighted_matrix: Matriz ponderada
            positive_ideal: Solução ideal positiva
            negative_ideal: Solução ideal negativa
            distances_positive: Distâncias para solução positiva
            distances_negative: Distâncias para solução negativa
            closeness_coefficients: Coeficientes de proximidade
            
        Returns:
            Dicionário com métricas
        """
        # Variância dos coeficientes de proximidade
        coefficient_variance = np.var(closeness_coefficients)
        
        # Diferença entre melhor e pior alternativa
        best_worst_difference = np.max(closeness_coefficients) - np.min(closeness_coefficients)
        
        # Entropia dos coeficientes
        coefficient_entropy = -np.sum(closeness_coefficients * np.log(closeness_coefficients + 1e-10))
        
        # Coeficiente de variação
        coefficient_cv = np.std(closeness_coefficients) / (np.mean(closeness_coefficients) + 1e-10)
        
        # Distância média para solução ideal positiva
        avg_distance_positive = np.mean(distances_positive)
        
        # Distância média para solução ideal negativa
        avg_distance_negative = np.mean(distances_negative)
        
        # Razão de separação (quão bem separadas estão as alternativas)
        separation_ratio = best_worst_difference / (np.mean(closeness_coefficients) + 1e-10)
        
        return {
            "coefficient_variance": coefficient_variance,
            "best_worst_difference": best_worst_difference,
            "coefficient_entropy": coefficient_entropy,
            "coefficient_coefficient_of_variation": coefficient_cv,
            "avg_distance_positive": avg_distance_positive,
            "avg_distance_negative": avg_distance_negative,
            "separation_ratio": separation_ratio,
            "num_alternatives": len(closeness_coefficients),
            "num_criteria": len(positive_ideal)
        }
    
    def calculate_sensitivity_analysis(self, 
                                     decision_matrix: np.ndarray,
                                     criteria_weights: List[float],
                                     criteria_types: List[str],
                                     weight_changes: List[float] = None) -> Dict[str, Any]:
        """
        Realiza análise de sensibilidade do método TOPSIS.
        
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
        original_scores = original_result["scores"]
        
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
                modified_scores = modified_result["scores"]
                
                # Calcular mudanças
                ranking_change = np.sum(np.abs(modified_ranking - original_ranking))
                score_change = np.mean(np.abs(modified_scores - original_scores))
                
                criterion_sensitivity[f"change_{change:.1f}"] = {
                    "ranking_change": ranking_change,
                    "score_change": score_change,
                    "modified_ranking": modified_ranking.tolist(),
                    "modified_scores": modified_scores.tolist(),
                    "modified_weights": modified_weights.tolist()
                }
            
            sensitivity_results[f"criterion_{i}"] = criterion_sensitivity
        
        return {
            "original_ranking": original_ranking.tolist(),
            "original_scores": original_scores.tolist(),
            "sensitivity_results": sensitivity_results
        }
    
    def get_ranking_explanation(self, 
                               decision_matrix: np.ndarray,
                               criteria_weights: List[float],
                               criteria_types: List[str],
                               alternative_index: int) -> Dict[str, Any]:
        """
        Fornece explicação detalhada do ranking de uma alternativa.
        
        Args:
            decision_matrix: Matriz de decisão
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios
            alternative_index: Índice da alternativa
            
        Returns:
            Explicação detalhada
        """
        result = self.evaluate(decision_matrix, criteria_weights, criteria_types)
        
        alternative_score = result["scores"][alternative_index]
        alternative_rank = result["ranks"][alternative_index]
        distance_positive = result["distances_positive"][alternative_index]
        distance_negative = result["distances_negative"][alternative_index]
        
        # Contribuição de cada critério
        weighted_values = result["weighted_matrix"][alternative_index, :]
        positive_ideal = result["positive_ideal"]
        negative_ideal = result["negative_ideal"]
        
        criterion_contributions = []
        for j in range(len(criteria_weights)):
            # Contribuição para distância positiva
            pos_contrib = (weighted_values[j] - positive_ideal[j]) ** 2
            # Contribuição para distância negativa
            neg_contrib = (weighted_values[j] - negative_ideal[j]) ** 2
            
            criterion_contributions.append({
                "criterion": j,
                "weighted_value": weighted_values[j],
                "positive_ideal": positive_ideal[j],
                "negative_ideal": negative_ideal[j],
                "positive_contribution": pos_contrib,
                "negative_contribution": neg_contrib
            })
        
        return {
            "alternative_index": alternative_index,
            "rank": alternative_rank,
            "closeness_coefficient": alternative_score,
            "distance_to_positive_ideal": distance_positive,
            "distance_to_negative_ideal": distance_negative,
            "criterion_contributions": criterion_contributions,
            "explanation": f"A alternativa {alternative_index} está na posição {alternative_rank + 1} "
                          f"com coeficiente de proximidade de {alternative_score:.4f}. "
                          f"Está a {distance_positive:.4f} da solução ideal positiva e "
                          f"a {distance_negative:.4f} da solução ideal negativa."
        }
    
    def compare_with_ideal_solutions(self, 
                                   decision_matrix: np.ndarray,
                                   criteria_weights: List[float],
                                   criteria_types: List[str]) -> Dict[str, Any]:
        """
        Compara todas as alternativas com as soluções ideais.
        
        Args:
            decision_matrix: Matriz de decisão
            criteria_weights: Pesos dos critérios
            criteria_types: Tipos dos critérios
            
        Returns:
            Comparação detalhada com soluções ideais
        """
        result = self.evaluate(decision_matrix, criteria_weights, criteria_types)
        
        positive_ideal = result["positive_ideal"]
        negative_ideal = result["negative_ideal"]
        weighted_matrix = result["weighted_matrix"]
        
        comparisons = []
        for i in range(len(decision_matrix)):
            alternative = weighted_matrix[i, :]
            
            # Calcular similaridade com cada solução ideal
            similarity_positive = 1.0 / (1.0 + result["distances_positive"][i])
            similarity_negative = 1.0 / (1.0 + result["distances_negative"][i])
            
            # Identificar critérios onde a alternativa está próxima da solução ideal
            positive_criteria = []
            negative_criteria = []
            
            for j in range(len(criteria_weights)):
                if abs(alternative[j] - positive_ideal[j]) < 0.1:
                    positive_criteria.append(j)
                if abs(alternative[j] - negative_ideal[j]) < 0.1:
                    negative_criteria.append(j)
            
            comparisons.append({
                "alternative": i,
                "similarity_positive": similarity_positive,
                "similarity_negative": similarity_negative,
                "positive_criteria": positive_criteria,
                "negative_criteria": negative_criteria,
                "closeness_coefficient": result["scores"][i],
                "rank": result["ranks"][i]
            })
        
        return {
            "positive_ideal": positive_ideal.tolist(),
            "negative_ideal": negative_ideal.tolist(),
            "comparisons": comparisons
        }

if __name__ == "__main__":
    
    # Teste simples do TOPSIS com uma matriz simulada de métricas de ML
    # Alternativas (linhas): Modelos A, B, C, D
    # Critérios (colunas): [top1_acc, precision_macro, recall_macro, f1_macro, total_params, avg_inference_time, memory_used_mb, gflops]
    decision_matrix = np.array([
        [0.85, 0.86, 0.84, 0.83, 0.835, 5.2e6, 0.008, 220.0, 2.4],   # Modelo A
        [0.84, 0.88, 0.86, 0.85, 0.855, 7.8e6, 0.010, 300.0, 3.1],   # Modelo B
        [0.80, 0.84, 0.82, 0.81, 0.815, 3.5e6, 0.006, 180.0, 1.8],   # Modelo C
        [0.84, 0.87, 0.85, 0.84, 0.845, 6.1e6, 0.007, 210.0, 2.2],   # Modelo D
    ], dtype=float)

    # Pesos dos critérios (devem somar 1 idealmente)
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

    # Tipos dos critérios
    criteria_types = [
        "benefit",  # top1_acc
        "benefit",  # top5_acc
        "benefit",  # precision_macro
        "benefit",  # recall_macro
        "benefit",  # f1_macro
        "cost",     # total_params
        "cost",     # avg_inference_time
        "cost",     # memory_used_mb
        "cost",     # gflops
    ]

    # Instancia TOPSIS e avalia
    topsis = TOPSIS(
        criteria_weights=criteria_weights,
        criteria_types=criteria_types,
        config={
            "normalization_method": "vector",
            "distance_metric": "euclidean",
            "weight_method": "ahp"  # usa pesos como fornecidos
        }
    )

    result = topsis.evaluate(
        decision_matrix=decision_matrix,
        criteria_weights=criteria_weights,
        criteria_types=criteria_types
    )

    # Exibe resultados do teste
    print("\n" + "="*60)
    print("TESTE TOPSIS - Ranking de Modelos de ML")
    print("="*60)
    
    model_names = ["Modelo A", "Modelo B", "Modelo C", "Modelo D"]
    ranks = result["ranks"]
    scores = result["scores"]
    sorted_idx = result["sorted_indices"]
    
    print("\n📊 RANKING FINAL:")
    print("-" * 60)
    for pos, idx in enumerate(sorted_idx, start=1):
        rank = ranks[idx]
        score = scores[idx]
        print(f"{pos}º lugar: {model_names[idx]:12s} | Score TOPSIS: {score:.6f} | Posição: {rank + 1}")

    print("\n📈 DISTÂNCIAS PARA SOLUÇÕES IDEAIS:")
    print("-" * 60)
    for i, name in enumerate(model_names):
        dist_pos = result["distances_positive"][i]
        dist_neg = result["distances_negative"][i]
        print(f"{name:12s} | Distância Ideal Positiva: {dist_pos:.6f} | Distância Ideal Negativa: {dist_neg:.6f}")

    print("\n🎯 SOLUÇÕES IDEAIS (ponderadas):")
    print("-" * 60)
    print(f"Positiva: {result['positive_ideal']}")
    print(f"Negativa: {result['negative_ideal']}")
    
    print("\n📋 MÉTRICAS ADICIONAIS:")
    print("-" * 60)
    metrics = result["metrics"]
    print(f"Variância dos coeficientes: {metrics['coefficient_variance']:.6f}")
    print(f"Diferença melhor-pior: {metrics['best_worst_difference']:.6f}")
    print(f"Razão de separação: {metrics['separation_ratio']:.6f}")
    print(f"Distância média à solução positiva: {metrics['avg_distance_positive']:.6f}")
    print(f"Distância média à solução negativa: {metrics['avg_distance_negative']:.6f}")
    
    print("\n✅ Teste TOPSIS concluído com sucesso!")
    print("="*60) 