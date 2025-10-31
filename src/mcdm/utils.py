"""
Utilitários para integração MCDM-PSO
====================================

Funções auxiliares para construção de matriz de decisão e conversão de scores.
"""

import numpy as np
from typing import List, Dict
from typing import List, Dict, Any, Optional, Tuple
import logging

# Ordem padrão dos critérios na matriz de decisão
DEFAULT_CRITERIA_ORDER = [
    "top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro",
    "total_params", "avg_inference_time", "memory_used_mb", "gflops"
]


def build_decision_matrix(all_metrics: List[Dict[str, float]], 
                         criteria_order: List[str] = None) -> np.ndarray:
    """
    Constrói matriz de decisão a partir de lista de métricas.
    
    Args:
        all_metrics: Lista de dicionários com métricas de cada partícula
        criteria_order: Ordem dos critérios na matriz (padrão: DEFAULT_CRITERIA_ORDER)
    
    Returns:
        Matriz NumPy de shape (n_particles, n_criteria)
    """
    if criteria_order is None:
        criteria_order = DEFAULT_CRITERIA_ORDER
    
    n_particles = len(all_metrics)
    n_criteria = len(criteria_order)
    
    decision_matrix = np.zeros((n_particles, n_criteria), dtype=float)
    
    for i, metrics in enumerate(all_metrics):
        for j, key in enumerate(criteria_order):
            decision_matrix[i, j] = metrics.get(key, 0.0)
    
    return decision_matrix


def mcdm_to_fitness(mcdm_scores: np.ndarray, 
                    invert: bool = True) -> np.ndarray:
    """
    Converte scores do MCDM para fitness do PSO.
    
    Args:
        mcdm_scores: Array com scores do MCDM (maior = melhor)
        invert: Se True, inverte scores (fitness = 1.0 - score). 
                Se False, usa score diretamente (para MCDMs que já minimizam)
    
    Returns:
        Array com valores de fitness (menor = melhor para PSO)
    """
    if invert:
        # MCDMs retornam scores onde maior = melhor
        # PSO minimiza, então invertemos
        fitness = 1.0 - mcdm_scores
    else:
        # Alguns MCDMs já retornam scores para minimização
        fitness = mcdm_scores.copy()
    
    return fitness


def validate_metrics(metrics: Dict[str, float], 
                     required_keys: List[str] = None) -> bool:
    """
    Valida se as métricas contêm as chaves necessárias.
    
    Args:
        metrics: Dicionário com métricas
        required_keys: Lista de chaves obrigatórias (padrão: DEFAULT_CRITERIA_ORDER)
    
    Returns:
        True se todas as chaves estão presentes
    """
    if required_keys is None:
        required_keys = DEFAULT_CRITERIA_ORDER
    
    missing_keys = [key for key in required_keys if key not in metrics]
    
    if missing_keys:
        return False
    
    return True


def extract_criteria_keys(criteria_order: List[str] = None) -> List[str]:
    """
    Retorna lista de chaves de critérios.
    
    Args:
        criteria_order: Ordem personalizada (padrão: DEFAULT_CRITERIA_ORDER)
    
    Returns:
        Lista de chaves de critérios
    """
    if criteria_order is None:
        return DEFAULT_CRITERIA_ORDER.copy()
    
    return criteria_order.copy()


def normalize_matrix(matrix: np.ndarray, 
                    method: str = "min_max",
                    criteria_types: Optional[List[str]] = None) -> np.ndarray:
    """
    Normaliza uma matriz de decisão.
    
    Args:
        matrix: Matriz de decisão (alternativas x critérios)
        method: Método de normalização ("min_max", "vector", "linear")
        criteria_types: Lista com tipos de critério ("benefit" ou "cost")
        
    Returns:
        Matriz normalizada
    """
    if criteria_types is None:
        criteria_types = ["benefit"] * matrix.shape[1]
    
    normalized_matrix = matrix.copy().astype(float)
    
    for j in range(matrix.shape[1]):
        criterion_type = criteria_types[j]
        column = matrix[:, j]
        
        if method == "min_max":
            min_val = np.min(column)
            max_val = np.max(column)
            
            if max_val != min_val:
                if criterion_type == "benefit":
                    normalized_matrix[:, j] = (column - min_val) / (max_val - min_val)
                else:  # cost
                    normalized_matrix[:, j] = (max_val - column) / (max_val - min_val)
            else:
                normalized_matrix[:, j] = 1.0
                
        elif method == "vector":
            # Normalização vector: r_ij = x_ij / sqrt(sum(x_ij^2))
            # No TOPSIS clássico, esta normalização não considera benefit/cost diretamente
            # Os tipos de critério são usados apenas na identificação das soluções ideais
            norm = np.sqrt(np.sum(column ** 2))
            if norm != 0:
                normalized_matrix[:, j] = column / norm
            else:
                normalized_matrix[:, j] = 0.0
                
        elif method == "linear":
            sum_val = np.sum(column)
            if sum_val != 0:
                normalized_matrix[:, j] = column / sum_val
            else:
                normalized_matrix[:, j] = 0.0
    
    return normalized_matrix


def calculate_weights(weights: List[float], 
                     method: str = "equal") -> np.ndarray:
    """
    Calcula pesos para os critérios.
    
    Args:
        weights: Lista de pesos ou configuração
        method: Método de cálculo ("equal", "ahp", "entropy")
        
    Returns:
        Array de pesos normalizados
    """
    if method == "equal":
        num_criteria = len(weights) if isinstance(weights, list) else weights
        return np.ones(num_criteria) / num_criteria
    
    elif method == "ahp":
        # Implementação simplificada do AHP
        if isinstance(weights, list):
            weights_array = np.array(weights)
            return weights_array / np.sum(weights_array)
        else:
            return np.ones(weights) / weights
    
    elif method == "entropy":
        # TODO: Implementar cálculo de pesos por entropia
        return np.ones(len(weights)) / len(weights)
    
    else:
        raise ValueError(f"Método de cálculo de pesos '{method}' não suportado")


def calculate_distance(point1: np.ndarray, 
                      point2: np.ndarray, 
                      method: str = "euclidean") -> float:
    """
    Calcula distância entre dois pontos.
    
    Args:
        point1: Primeiro ponto
        point2: Segundo ponto
        method: Método de distância ("euclidean", "manhattan", "chebyshev")
        
    Returns:
        Distância calculada
    """
    if method == "euclidean":
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    elif method == "manhattan":
        return np.sum(np.abs(point1 - point2))
    
    elif method == "chebyshev":
        return np.max(np.abs(point1 - point2))
    
    else:
        raise ValueError(f"Método de distância '{method}' não suportado")


def find_ideal_solutions(matrix: np.ndarray, 
                        criteria_types: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encontra soluções ideais positiva e negativa.
    
    Args:
        matrix: Matriz de decisão normalizada
        criteria_types: Tipos de critério ("benefit" ou "cost")
        
    Returns:
        Tupla com (solução_ideal_positiva, solução_ideal_negativa)
    """
    positive_ideal = np.zeros(matrix.shape[1])
    negative_ideal = np.zeros(matrix.shape[1])
    
    for j in range(matrix.shape[1]):
        criterion_type = criteria_types[j]
        column = matrix[:, j]
        
        if criterion_type == "benefit":
            positive_ideal[j] = np.max(column)
            negative_ideal[j] = np.min(column)
        else:  # cost
            positive_ideal[j] = np.min(column)
            negative_ideal[j] = np.max(column)
    
    return positive_ideal, negative_ideal


def calculate_consistency_ratio(comparison_matrix: np.ndarray) -> float:
    """
    Calcula a razão de consistência para matrizes de comparação AHP.
    
    Args:
        comparison_matrix: Matriz de comparação pareada
        
    Returns:
        Razão de consistência
    """
    # Calcular autovalor principal
    eigenvalues = np.linalg.eigvals(comparison_matrix)
    max_eigenvalue = np.max(np.real(eigenvalues))
    
    # Calcular índice de consistência
    n = comparison_matrix.shape[0]
    consistency_index = (max_eigenvalue - n) / (n - 1)
    
    # Índices aleatórios para diferentes tamanhos de matriz
    random_indices = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    
    random_index = random_indices.get(n, 1.49)
    
    # Calcular razão de consistência
    consistency_ratio = consistency_index / random_index
    
    return consistency_ratio


def validate_weights(weights: np.ndarray, 
                    tolerance: float = 1e-6) -> bool:
    """
    Valida se os pesos estão corretamente normalizados.
    
    Args:
        weights: Array de pesos
        tolerance: Tolerância para validação
        
    Returns:
        True se os pesos são válidos
    """
    # Verificar se soma é aproximadamente 1
    weight_sum = np.sum(weights)
    if abs(weight_sum - 1.0) > tolerance:
        return False
    
    # Verificar se todos os pesos são não-negativos
    if np.any(weights < 0):
        return False
    
    return True


def aggregate_criteria(alternatives: np.ndarray, 
                      weights: np.ndarray,
                      method: str = "weighted_sum") -> np.ndarray:
    """
    Agrega critérios usando diferentes métodos.
    
    Args:
        alternatives: Matriz de alternativas normalizada
        weights: Pesos dos critérios
        method: Método de agregação ("weighted_sum", "weighted_product")
        
    Returns:
        Array com scores agregados
    """
    if method == "weighted_sum":
        return np.sum(alternatives * weights, axis=1)
    
    elif method == "weighted_product":
        # Evitar zeros na multiplicação
        alternatives_safe = np.where(alternatives == 0, 1e-10, alternatives)
        return np.prod(alternatives_safe ** weights, axis=1)
    
    else:
        raise ValueError(f"Método de agregação '{method}' não suportado")


def rank_alternatives(scores: np.ndarray, 
                     ascending: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rankeia alternativas baseado nos scores.
    
    Args:
        scores: Array com scores das alternativas
        ascending: Se deve ordenar em ordem crescente
        
    Returns:
        Tupla com (ranks, sorted_indices)
    """
    if ascending:
        sorted_indices = np.argsort(scores)
    else:
        sorted_indices = np.argsort(scores)[::-1]
    
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(scores))
    
    return ranks, sorted_indices


def calculate_sensitivity(alternatives: np.ndarray,
                         weights: np.ndarray,
                         criterion_index: int,
                         weight_change: float = 0.1) -> Dict[str, float]:
    """
    Calcula sensibilidade de um critério específico.
    
    Args:
        alternatives: Matriz de alternativas
        weights: Pesos originais
        criterion_index: Índice do critério a analisar
        weight_change: Mudança percentual no peso
        
    Returns:
        Dicionário com métricas de sensibilidade
    """
    original_weights = weights.copy()
    
    # Calcular score original
    original_scores = aggregate_criteria(alternatives, original_weights)
    original_ranking = rank_alternatives(original_scores)[0]
    
    # Modificar peso do critério
    modified_weights = original_weights.copy()
    modified_weights[criterion_index] *= (1 + weight_change)
    modified_weights = modified_weights / np.sum(modified_weights)  # Renormalizar
    
    # Calcular novo score
    modified_scores = aggregate_criteria(alternatives, modified_weights)
    modified_ranking = rank_alternatives(modified_scores)[0]
    
    # Calcular métricas de sensibilidade
    ranking_change = np.sum(np.abs(modified_ranking - original_ranking))
    score_change = np.mean(np.abs(modified_scores - original_scores))
    
    return {
        "ranking_change": ranking_change,
        "score_change": score_change,
        "original_ranking": original_ranking,
        "modified_ranking": modified_ranking
    }
