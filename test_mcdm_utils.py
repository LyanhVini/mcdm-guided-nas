"""
Teste das Funções Auxiliares MCDM
=================================

Valida build_decision_matrix e mcdm_to_fitness.
"""

import numpy as np
import sys
import os
import importlib.util

# Importa diretamente
src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_dir)

utils_path = os.path.join(src_dir, "mcdm", "utils.py")
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

build_decision_matrix = utils_module.build_decision_matrix
mcdm_to_fitness = utils_module.mcdm_to_fitness
validate_metrics = utils_module.validate_metrics


def test_build_decision_matrix():
    """Testa construção de matriz de decisão."""
    print("="*70)
    print("TESTE: build_decision_matrix")
    print("="*70)
    
    # Cria métricas de teste
    all_metrics = [
        {
            "top1_acc": 0.85,
            "top5_acc": 0.92,
            "precision_macro": 0.88,
            "recall_macro": 0.86,
            "f1_macro": 0.87,
            "total_params": 3000000,
            "avg_inference_time": 0.010,
            "memory_used_mb": 200,
            "gflops": 2.0
        },
        {
            "top1_acc": 0.80,
            "top5_acc": 0.90,
            "precision_macro": 0.85,
            "recall_macro": 0.82,
            "f1_macro": 0.83,
            "total_params": 5000000,
            "avg_inference_time": 0.012,
            "memory_used_mb": 250,
            "gflops": 2.5
        }
    ]
    
    # Constrói matriz
    matrix = build_decision_matrix(all_metrics)
    
    print(f"Matriz construida:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Esperado: (2, 9)")
    
    # Validações
    assert matrix.shape == (2, 9), f"Shape errado: {matrix.shape}"
    assert matrix[0, 0] == 0.85, f"Valor incorreto: {matrix[0, 0]}"
    assert matrix[0, 5] == 3000000, f"Valor incorreto: {matrix[0, 5]}"
    assert matrix[1, 0] == 0.80, f"Valor incorreto: {matrix[1, 0]}"
    
    print(f"  [OK] Valores corretos")
    print(f"  Primeira linha: {matrix[0, :]}")
    
    # Testa com métricas faltantes
    incomplete_metrics = [
        {"top1_acc": 0.85, "top5_acc": 0.92}  # Faltam outras métricas
    ]
    matrix_incomplete = build_decision_matrix(incomplete_metrics)
    assert matrix_incomplete.shape == (1, 9), f"Shape errado: {matrix_incomplete.shape}"
    assert matrix_incomplete[0, 0] == 0.85
    assert matrix_incomplete[0, 2] == 0.0  # Valores faltantes = 0.0
    
    print(f"  [OK] Trata metricas faltantes corretamente")
    
    return True


def test_mcdm_to_fitness():
    """Testa conversão de scores MCDM para fitness."""
    print("\n" + "="*70)
    print("TESTE: mcdm_to_fitness")
    print("="*70)
    
    # MCDM scores (maior = melhor)
    mcdm_scores = np.array([0.9, 0.7, 0.8, 0.6])
    
    # Converte para fitness (menor = melhor)
    fitness = mcdm_to_fitness(mcdm_scores, invert=True)
    
    print(f"MCDM scores (maior=melhor): {mcdm_scores}")
    print(f"Fitness scores (menor=melhor): {fitness}")
    
    # Validações
    assert len(fitness) == len(mcdm_scores), "Tamanho diferente"
    assert np.allclose(fitness, 1.0 - mcdm_scores), "Conversao incorreta"
    
    # Melhor MCDM score (0.9) deve ter menor fitness (0.1)
    best_mcdm_idx = np.argmax(mcdm_scores)
    best_fitness_idx = np.argmin(fitness)
    assert best_mcdm_idx == best_fitness_idx, "Ranking invertido incorretamente"
    
    print(f"  [OK] Conversao correta")
    print(f"  Melhor MCDM (idx {best_mcdm_idx}, score {mcdm_scores[best_mcdm_idx]})")
    print(f"  Melhor Fitness (idx {best_fitness_idx}, fitness {fitness[best_fitness_idx]})")
    
    # Testa sem inverter
    fitness_no_invert = mcdm_to_fitness(mcdm_scores, invert=False)
    assert np.allclose(fitness_no_invert, mcdm_scores), "Sem inversao deve manter valores"
    
    print(f"  [OK] Modo sem inversao funciona")
    
    return True


def test_validate_metrics():
    """Testa validação de métricas."""
    print("\n" + "="*70)
    print("TESTE: validate_metrics")
    print("="*70)
    
    # Métricas completas
    complete_metrics = {
        "top1_acc": 0.85,
        "top5_acc": 0.92,
        "precision_macro": 0.88,
        "recall_macro": 0.86,
        "f1_macro": 0.87,
        "total_params": 3000000,
        "avg_inference_time": 0.010,
        "memory_used_mb": 200,
        "gflops": 2.0
    }
    
    is_valid = validate_metrics(complete_metrics)
    assert is_valid, "Metricas completas devem ser validas"
    print(f"  [OK] Metricas completas sao validas")
    
    # Métricas incompletas
    incomplete_metrics = {"top1_acc": 0.85}  # Faltam outras
    
    is_valid = validate_metrics(incomplete_metrics)
    assert not is_valid, "Metricas incompletas devem ser invalidas"
    print(f"  [OK] Metricas incompletas sao detectadas")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SUITE DE TESTES - Funcoes Auxiliares MCDM")
    print("="*70)
    
    results = []
    
    try:
        results.append(("build_decision_matrix", test_build_decision_matrix()))
    except Exception as e:
        print(f"\n[ERRO] ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("build_decision_matrix", False))
    
    try:
        results.append(("mcdm_to_fitness", test_mcdm_to_fitness()))
    except Exception as e:
        print(f"\n[ERRO] ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("mcdm_to_fitness", False))
    
    try:
        results.append(("validate_metrics", test_validate_metrics()))
    except Exception as e:
        print(f"\n[ERRO] ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("validate_metrics", False))
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[OK] PASSOU" if result else "[ERRO] FALHOU"
        print(f"{name:30s}: {status}")
    
    print(f"\nTestes passados: {passed}/{total}")
    
    if passed == total:
        print("\n[OK] TODOS OS TESTES PASSARAM!")
    else:
        print(f"\n[AVISO] {total - passed} TESTE(S) FALHARAM")
    
    print("="*70)

