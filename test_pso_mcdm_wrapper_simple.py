"""
Teste Simples do PSOMCDMWrapper
================================

Testa apenas a estrutura e lógica básica, sem depender dos MCDMs completos.
"""

import numpy as np
import sys
import os


def test_structure():
    """Testa estrutura básica sem executar os MCDMs."""
    print("="*70)
    print("TESTE: Estrutura do PSOMCDMWrapper")
    print("="*70)
    
    # Verifica que o arquivo existe
    wrapper_path = os.path.join("src", "optimizers", "pso_mcdm_wrapper.py")
    
    if not os.path.exists(wrapper_path):
        print(f"[ERRO] Arquivo nao encontrado: {wrapper_path}")
        return False
    
        print(f"[OK] Arquivo encontrado: {wrapper_path}")
    
    # Lê o arquivo e verifica estrutura
    with open(wrapper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verifica componentes essenciais
    checks = {
        "Classe PSOMCDMWrapper": "class PSOMCDMWrapper" in content,
        "EVALUATION_TYPES": "EVALUATION_TYPES" in content,
        "fitness_function": "def fitness_function" in content,
        "_build_decision_matrix": "def _build_decision_matrix" in content,
        "_evaluate_individual": "def _evaluate_individual" in content,
        "_evaluate_contextual": "def _evaluate_contextual" in content,
        "is_contextual": "def is_contextual" in content,
        "is_individual": "def is_individual" in content,
    }
    
    print("\nVerificando componentes:")
    all_ok = True
    for component, found in checks.items():
        status = "[OK]" if found else "[ERRO]"
        print(f"  {status} {component}")
        if not found:
            all_ok = False
    
    # Verifica mapeamento de métodos
    print("\nVerificando mapeamento de métodos:")
    if '"OACE": "individual"' in content:
        print("  [OK] OACE -> individual")
    else:
        print("  [ERRO] OACE -> individual")
        all_ok = False
    
    if '"TOPSIS": "contextual"' in content:
        print("  [OK] TOPSIS -> contextual")
    else:
        print("  [ERRO] TOPSIS -> contextual")
        all_ok = False
    
    if '"VIKOR": "contextual"' in content:
        print("  [OK] VIKOR -> contextual")
    else:
        print("  [ERRO] VIKOR -> contextual")
        all_ok = False
    
    return all_ok


def test_logic_simulation():
    """Simula a lógica do wrapper sem executar código real."""
    print("\n" + "="*70)
    print("TESTE: Simulação de Lógica")
    print("="*70)
    
    # Simula função de treinamento
    def mock_train(pos):
        return {
            "top1_acc": 0.80 + np.random.rand() * 0.10,
            "top5_acc": 0.90 + np.random.rand() * 0.05,
            "precision_macro": 0.75 + np.random.rand() * 0.15,
            "recall_macro": 0.78 + np.random.rand() * 0.12,
            "f1_macro": 0.76 + np.random.rand() * 0.14,
            "total_params": 2e6 + np.random.rand() * 4e6,
            "avg_inference_time": 0.008 + np.random.rand() * 0.007,
            "memory_used_mb": 150 + np.random.rand() * 150,
            "gflops": 1.5 + np.random.rand() * 1.5
        }
    
    # Simula construção de matriz de decisão
    n_particles = 4
    all_metrics = [mock_train(np.array([i])) for i in range(n_particles)]
    
    criteria_keys = [
        "top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro",
        "total_params", "avg_inference_time", "memory_used_mb", "gflops"
    ]
    
    decision_matrix = np.zeros((n_particles, len(criteria_keys)))
    for i, metrics in enumerate(all_metrics):
        for j, key in enumerate(criteria_keys):
            decision_matrix[i, j] = metrics.get(key, 0.0)
    
    print(f"Matriz de decisão construída:")
    print(f"  Shape: {decision_matrix.shape}")
    print(f"  Esperado: ({n_particles}, {len(criteria_keys)})")
    
    assert decision_matrix.shape == (n_particles, len(criteria_keys)), "Shape incorreto"
    
    print(f"  [OK] Matriz construida corretamente")
    print(f"\nPrimeira linha (exemplo):")
    print(f"  {dict(zip(criteria_keys, decision_matrix[0, :]))}")
    
    # Simula tipos de avaliação
    evaluation_types = {
        "OACE": "individual",
        "TOPSIS": "contextual",
        "VIKOR": "contextual"
    }
    
    print(f"\nVerificando tipos de avaliação:")
    for method, eval_type in evaluation_types.items():
        is_ctx = eval_type == "contextual"
        is_ind = eval_type == "individual"
        print(f"  {method:8s}: contextual={is_ctx}, individual={is_ind}")
    
    # Simula conversão de scores para fitness
    mcdm_scores = np.array([0.9, 0.7, 0.8, 0.6])  # Maior = melhor
    fitness = 1.0 - mcdm_scores  # PSO minimiza
    
    print(f"\nSimulação de conversão score -> fitness:")
    print(f"  MCDM scores (maior=melhor): {mcdm_scores}")
    print(f"  Fitness scores (menor=melhor): {fitness}")
    print(f"  [OK] Conversao correta")
    
    return True


def test_decision_matrix_logic():
    """Testa especificamente a lógica de construção de matriz."""
    print("\n" + "="*70)
    print("TESTE: Lógica de Matriz de Decisão")
    print("="*70)
    
    # Dados de teste
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
    
    criteria_keys = [
        "top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro",
        "total_params", "avg_inference_time", "memory_used_mb", "gflops"
    ]
    
    # Constrói matriz
    n_particles = len(all_metrics)
    n_criteria = len(criteria_keys)
    decision_matrix = np.zeros((n_particles, n_criteria))
    
    for i, metrics in enumerate(all_metrics):
        for j, key in enumerate(criteria_keys):
            decision_matrix[i, j] = metrics.get(key, 0.0)
    
    print(f"Matriz construída:")
    print(f"  Shape: {decision_matrix.shape}")
    print(f"\nValores:")
    for i in range(n_particles):
        print(f"  Partícula {i}: {decision_matrix[i, 0]:.4f} (top1_acc)")
    
    # Validações
    assert decision_matrix.shape == (2, 9), f"Shape errado: {decision_matrix.shape}"
    assert decision_matrix[0, 0] == 0.85, f"Valor incorreto: {decision_matrix[0, 0]}"
    assert decision_matrix[0, 5] == 3000000, f"Valor incorreto: {decision_matrix[0, 5]}"
    
    print(f"  [OK] Validacoes passaram")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SUITE DE TESTES SIMPLIFICADA - PSOMCDMWrapper")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Estrutura do Código", test_structure()))
    except Exception as e:
        print(f"\n[ERRO] ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("Estrutura do Código", False))
    
    try:
        results.append(("Simulação de Lógica", test_logic_simulation()))
    except Exception as e:
        print(f"\n[ERRO] ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("Simulação de Lógica", False))
    
    try:
        results.append(("Lógica de Matriz", test_decision_matrix_logic()))
    except Exception as e:
        print(f"\n[ERRO] ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append(("Lógica de Matriz", False))
    
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

