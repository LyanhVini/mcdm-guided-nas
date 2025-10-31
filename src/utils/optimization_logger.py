"""
Logger de Otimização
===================

Fornece visibilidade completa do processo de otimização,
facilitando debugging e análise de performance.
"""
import numpy as np
from typing import Dict, Any, Tuple, List, Type

def _print_header(self, title: str, width: int = 80):
    """Imprime um cabeçalho formatado para seções importantes"""
    print("\n" + "="*width)
    print(f"🚀 {title}")
    print("="*width)

def _print_section(self, title: str, width: int = 60):
    """Imprime uma seção formatada para subseções"""
    print(f"\n{'─'*width}")
    print(f"📋 {title}")
    print(f"{'─'*width}")

def _print_step(self, step: str, details: str = ""):
    """Imprime um passo do algoritmo com formatação consistente"""
    print(f"\n→ {step}")
    if details:
        print(f"   {details}")

def _print_configuration(self):
    """Imprime a configuração completa do algoritmo"""
    self._print_section("CONFIGURAÇÃO DO ALGORITMO")
    
    print(f"🎯 Parâmetros Gerais:")
    print(f"   • Tamanho da população: {self.population_size}")
    print(f"   • Máximo de iterações: {self.max_iter}")
    print(f"   • Parâmetro λ (trade-off): {self.lambda_param}")
    print(f"   • Dimensões do espaço: {self.n_dim}")
    
    print(f"\n🏗️  Arquiteturas Disponíveis:")
    for i, arch in enumerate(self.architectures_to_optimize, 1):
        print(f"   {i}. {arch}")
    
    print(f"\n🐟 Parâmetros AFSA:")
    for key, value in self.afsa_params.items():
        print(f"   • {key}: {value}")
    
    print(f"\n🌊 Parâmetros PSO:")
    for key, value in self.pso_params.items():
        print(f"   • {key}: {value}")
    
    print(f"\n📊 Limites dos Parâmetros:")
    for param, (min_val, max_val) in self.param_bounds.items():
        print(f"   • {param}: [{min_val}, {max_val}]")

def _print_iteration_header(self, phase: str, iteration: int, total_iterations: int = None):
    """Imprime cabeçalho de iteração com informações da fase"""
    if total_iterations:
        progress = f"({iteration}/{total_iterations})"
    else:
        progress = f"(#{iteration})"
    
    print(f"\n{'='*60}")
    print(f"🔄 FASE: {phase} {progress}")
    print(f"{'='*60}")

def _print_candidate_details(self, candidate: np.ndarray, metrics: Dict[str, float], 
                            architecture_name: str, architecture_params: Dict[str, Any], 
                            oace_score: float = None, candidate_id: int = None):
    """Imprime detalhes completos de um candidato"""
    if candidate_id is not None:
        print(f"\n📋 Candidato #{candidate_id}")
    else:
        print(f"\n📋 Detalhes do Candidato")
    
    print(f"   🏗️  Arquitetura: {architecture_name}")
    print(f"   ⚙️  Parâmetros: {architecture_params}")
    
    if oace_score is not None:
        print(f"   🎯 Score OACE: {oace_score:.6f}")
    
    print(f"   📊 Métricas de Assertividade:")
    print(f"      • Top-1 Accuracy: {metrics.get('top1_acc', 0):.4f}")
    print(f"      • Top-5 Accuracy: {metrics.get('top5_acc', 0):.4f}")
    print(f"      • Precision Macro: {metrics.get('precision_macro', 0):.4f}")
    print(f"      • Recall Macro: {metrics.get('recall_macro', 0):.4f}")
    print(f"      • F1 Macro: {metrics.get('f1_macro', 0):.4f}")
    
    print(f"   💰 Métricas de Custo:")
    print(f"      • Total Parâmetros: {metrics.get('total_params', 0):,}")
    print(f"      • Tempo Inferência: {metrics.get('avg_inference_time', 0):.4f}s")
    print(f"      • Memória Usada: {metrics.get('memory_used_mb', 0):.2f} MB")
    print(f"      • GFLOPs: {metrics.get('gflops', 0):.2f}")

def _print_population_summary(self, population: np.ndarray, fitness_values: np.ndarray, 
                                phase: str, iteration: int = None):
    """Imprime resumo da população atual"""
    if iteration is not None:
        print(f"\n📊 Resumo da População - {phase} (Iteração {iteration})")
    else:
        print(f"\n📊 Resumo da População - {phase}")
    
    print(f"   • Tamanho da população: {len(population)}")
    print(f"   • Melhor fitness: {np.max(fitness_values):.6f}")
    print(f"   • Pior fitness: {np.min(fitness_values):.6f}")
    print(f"   • Fitness médio: {np.mean(fitness_values):.6f}")
    print(f"   • Desvio padrão: {np.std(fitness_values):.6f}")
    
    # Mostra os 3 melhores candidatos
    best_indices = np.argsort(fitness_values)[-3:][::-1]
    print(f"   🏆 Top 3 Candidatos:")
    for i, idx in enumerate(best_indices, 1):
        arch_name, _ = self._convert_to_architecture_params(population[idx])
        print(f"      {i}. {arch_name} - Fitness: {fitness_values[idx]:.6f}")

def _print_cache_stats(self):
    """Imprime estatísticas do cache"""
    total_evaluations = self.cache_hits + self.cache_misses
    cache_efficiency = (self.cache_hits / total_evaluations * 100) if total_evaluations > 0 else 0
    
    print(f"\n💾 Estatísticas de Cache:")
    print(f"   • Total de avaliações: {total_evaluations}")
    print(f"   • Cache hits: {self.cache_hits} ({cache_efficiency:.1f}%)")
    print(f"   • Cache misses: {self.cache_misses}")
    print(f"   • Candidatos únicos avaliados: {len(self.candidates_cache)}")

def _print_phase_summary(self, phase: str, best_fitness: float, best_architecture: str, 
                        best_params: Dict[str, Any], total_time: float = None):
    """Imprime resumo de uma fase do algoritmo"""
    print(f"\n{'='*60}")
    print(f"✅ FASE {phase} CONCLUÍDA")
    print(f"{'='*60}")
    print(f"🏆 Melhor Resultado:")
    print(f"   • Arquitetura: {best_architecture}")
    print(f"   • Score OACE: {best_fitness:.6f}")
    print(f"   • Parâmetros: {best_params}")
    if total_time:
        print(f"   • Tempo total: {total_time:.2f}s")

def _print_final_results(self, best_architecture: str, best_params: Dict[str, Any], 
                        best_fitness: float, final_metrics: Dict[str, float]):
    """Imprime resultados finais formatados"""
    print(f"\n{'='*80}")
    print(f"🎉 OTIMIZAÇÃO HÍBRIDA AFSA-PSO CONCLUÍDA")
    print(f"{'='*80}")
    
    print(f"\n🏆 RESULTADO FINAL:")
    print(f"   • Melhor arquitetura: {best_architecture}")
    print(f"   • Score OACE final: {best_fitness:.6f}")
    print(f"   • Parâmetros da melhor arquitetura:")
    for key, value in best_params.items():
        print(f"     - {key}: {value}")
    
    print(f"\n📊 MÉTRICAS FINAIS DA MELHOR ARQUITETURA:")
    print(f"   • Top-1 Accuracy: {final_metrics['top1_acc']:.4f}")
    print(f"   • Top-5 Accuracy: {final_metrics['top5_acc']:.4f}")
    print(f"   • Precision Macro: {final_metrics['precision_macro']:.4f}")
    print(f"   • Recall Macro: {final_metrics['recall_macro']:.4f}")
    print(f"   • F1 Macro: {final_metrics['f1_macro']:.4f}")
    print(f"   • Total Parâmetros: {final_metrics['total_params']:,}")
    print(f"   • Tempo Inferência: {final_metrics['avg_inference_time']:.4f}s")
    print(f"   • Memória: {final_metrics['memory_used_mb']:.2f} MB")
    print(f"   • GFLOPs: {final_metrics['gflops']:.2f}")
    
    self._print_cache_stats()