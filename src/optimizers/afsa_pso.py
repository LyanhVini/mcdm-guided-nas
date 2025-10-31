"""
Otimizador Híbrido AFSA-PSO
===========================

Implementação do algoritmo híbrido que combina Artificial Fish Swarm Algorithm (AFSA)
com Particle Swarm Optimization (PSO) para Neural Architecture Search.

Funcionalidades:
- Combinação inteligente de AFSA (exploração global) e PSO (exploração local)
- Adaptação dinâmica dos parâmetros durante a otimização
- Mecanismos de convergência e detecção de estagnação
- Suporte a diferentes espaços de busca e funções de fitness
- Logging detalhado do processo de otimização

Este é o núcleo do sistema de otimização, responsável por encontrar
as melhores arquiteturas neurais através da busca híbrida.
"""
# source oace/bin/activate
# python3 -X utf8 -u -m optimizers.afsa_ga_pso 2>&1 | tee teste5.log
import numpy as np
from optimizers.afsa import AFSA
from optimizers.pso import PSO
from models.architecture_loader import archictectures
from utils.oace_evaluation import calculate_oace_score
from typing import Dict, Any, Tuple, List, Type
from pydantic import BaseModel
import torch
from tqdm import tqdm
import sys
import os   
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import get_cifar10_dataloaders
from utils.optimization_logger import *
import time

class AFSAPSO:
    """
    Implementa o algoritmo híbrido AFSA-PSO para otimização de arquiteturas
    de redes neurais profundas usando o score OACE como função de fitness.
    Otimiza tanto a escolha da arquitetura quanto seus parâmetros.
    """

    def __init__(
        self,
        population_size: int,
        max_iter: int,
        train_loader,
        val_loader,
        test_loader,
        classes: List[str],
        lambda_param: float = 0.5,
        afsa_params: Dict[str, Any] = None,
        pso_params: Dict[str, Any] = None,
        architectures_to_optimize: List[str] = None,
        log_dir: str = "results"
    ):
        """
        Inicializa o otimizador híbrido.

        Args:
        population_size (int): Tamanho da população para cada algoritmo.
        max_iter (int): Número máximo de iterações para cada algoritmo.
            train_loader: DataLoader para treinamento.
            val_loader: DataLoader para validação.
            test_loader: DataLoader para teste.
            classes (List[str]): Lista de classes do problema.
            lambda_param (float): Parâmetro de trade-off λ para o OACE (entre 0 e 1).
        afsa_params (dict): Parâmetros para o AFSA.
        pso_params (dict): Parâmetros para o PSO.
            architectures_to_optimize (List[str]): Lista de arquiteturas a otimizar. Se None, usa todas disponíveis.
            log_dir (str): Diretório para salvar os logs da otimização.
        """
        # Arquiteturas disponíveis para otimização
        if architectures_to_optimize is None:
            self.architectures_to_optimize = list(archictectures.keys())
        else:
            self.architectures_to_optimize = architectures_to_optimize

        print(f"📋 Arquiteturas para otimização: {self.architectures_to_optimize}")

        # Todas as arquiteturas e informações
        self.all_architectures = {
            name: archictectures[name] for name in self.architectures_to_optimize
        }
        self.population_size = population_size
        self.max_iter = max_iter
        self.lambda_param = lambda_param

        # DataLoaders e classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes = classes

        # Define os limites do espaço de busca considerando todas as arquiteturas
        self.param_bounds = self._get_unified_param_bounds()
        # +1 dimensão para escolha da arquitetura (architecture_index)
        self.n_dim = len(self.param_bounds) + 1
        
        print(f"🎯 Dimensões do espaço de busca: {self.n_dim}")
        print(f"📏 Limites dos parâmetros: {self.param_bounds}")
        
        # Parâmetros padrão para o AFSA
        if afsa_params is None:
            afsa_params = {"visual": 0.5, "step": 0.1, "try_times": 5, "max_iter": 50}
        self.afsa_params = afsa_params
        
        # Parâmetros padrão para o PSO (ajustados para mais diversidade)
        if pso_params is None:
            pso_params = {
                "c1": 1.5,    # Aumentado para mais exploração individual
                "c2": 1.0,    # Aumentado para mais exploração social
                "w": 0.7,     # Reduzido um pouco para mais controle
                "k": 3, 
                "p": 2
            }
        self.pso_params = pso_params
        
        # Inicialização dos componentes
        self.afsa = None
        self.pso = None
        self.best_solution = None
        self.best_fitness = float("-inf")  # O OACE é maximizado
        self.history = []

        # Métricas e limites
        self.metrics_history = []
        self.metrics_ranges = None
        
        # Cache para evitar re-avaliação de candidatos idênticos
        self.candidates_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Configuração inicial do experimento
        experiment_config = {
            "population_size": population_size,
            "max_iter": max_iter,
            "lambda_param": lambda_param,
            "afsa_params": afsa_params,
            "pso_params": pso_params,
            "architectures_to_optimize": architectures_to_optimize
        }
        

    def _get_unified_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define os limites unificados do espaço de busca para todos os parâmetros
        de todas as arquiteturas disponíveis.
        """
        bounds = {}
        all_param_names = set()

        # Coleta todos os parâmetros únicos de todas as arquiteturas
        for arch_name, arch_info in self.all_architectures.items():
            params_class = type(arch_info["params"])
            for field_name in params_class.model_fields.keys():
                if field_name not in [
                    "num_classes",
                    "weight_init_fn",
                    "batch_norm",
                    "randomize",
                ]:
                    all_param_names.add(field_name)

        # Define limites unificados que cobrem todas as arquiteturas
        for param_name in all_param_names:
            if param_name == "dropout_rate":
                bounds[param_name] = (0.0, 0.5)
            elif param_name == "min_channels":
                # Unifica os limites: min de todos os mínimos, max de todos os máximos
                bounds[param_name] = (
                    8,
                    64,
                )  # Cobre tanto CNN (8-64) quanto MobileNet (16-64)
            elif param_name == "max_channels":
                # Unifica os limites para cobrir todas as arquiteturas
                bounds[param_name] = (
                    128,
                    2048,
                )  # Cobre CNN (128-512) e MobileNet (512-2048)
            elif param_name == "num_layers":
                # Unifica os limites para cobrir todas as arquiteturas
                bounds[param_name] = (2, 20)  # Cobre CNN (2-8) e MobileNet (8-20)
            elif param_name == "width_multiplier":
                bounds[param_name] = (0.5, 1.5)
            elif param_name == "resolution_multiplier":
                bounds[param_name] = (0.5, 1.0)
            else:
                # Para parâmetros não mapeados, usa valores padrão
                bounds[param_name] = (0.0, 1.0)

        return bounds

    def _get_architecture_from_vector(
        self, x: np.ndarray
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extrai a arquitetura escolhida e seus parâmetros do vetor de otimização.

        Args:
            x (np.ndarray): Vetor de parâmetros completo [architecture_index, param1, param2, ...]

        Returns:
            Tuple[str, Dict[str, Any]]: (nome_da_arquitetura, parâmetros_da_arquitetura)
        """
        # A primeira dimensão é o índice da arquitetura (normalizado entre 0 e 1)
        architecture_index_normalized = x[0]

        # Converte para índice discreto
        n_architectures = len(self.architectures_to_optimize)
        architecture_index = int(architecture_index_normalized * n_architectures)

        # Garante que o índice está dentro dos limites
        architecture_index = max(0, min(architecture_index, n_architectures - 1))

        # Obtém o nome da arquitetura
        architecture_name = self.architectures_to_optimize[architecture_index]

        # Extrai os parâmetros (resto do vetor)
        params_vector = x[1:]

        # Converte parâmetros para a arquitetura específica
        architecture_params = self._convert_params_for_architecture(
            params_vector, architecture_name
        )

        return architecture_name, architecture_params

    def _convert_params_for_architecture(
        self, params_vector: np.ndarray, architecture_name: str
    ) -> Dict[str, Any]:
        """
        Converte um vetor de parâmetros para os parâmetros específicos de uma arquitetura.

        Args:
            params_vector (np.ndarray): Vetor de parâmetros normalizados [0,1]
            architecture_name (str): Nome da arquitetura

        Returns:
            Dict[str, Any]: Parâmetros da arquitetura
        """
        architecture_info = self.all_architectures[architecture_name]
        params_class = type(architecture_info["params"])

        params = {}

        # Converte cada parâmetro do vetor unificado para os parâmetros específicos da arquitetura
        param_index = 0
        for param_name, (min_val, max_val) in self.param_bounds.items():
            # Verifica se este parâmetro existe na arquitetura atual
            if param_name in params_class.model_fields:
                # Normaliza o valor para o intervalo [min_val, max_val]
                normalized_value = (
                    params_vector[param_index] * (max_val - min_val) + min_val
                )

                # Converte para o tipo correto baseado no campo da classe
                field = params_class.model_fields[param_name]
                field_type = field.annotation

                if field_type == int:
                    params[param_name] = int(round(normalized_value))
                elif field_type == float:
                    params[param_name] = float(normalized_value)
                elif field_type == bool:
                    params[param_name] = bool(round(normalized_value))
                else:
                    params[param_name] = normalized_value

            param_index += 1

        # Adiciona parâmetros fixos
        params["num_classes"] = len(self.classes)

        # Adiciona batch_norm como True por padrão se existir no modelo
        if "batch_norm" in params_class.model_fields:
            params["batch_norm"] = True

        return params

    def _convert_to_architecture_params(
        self, x: np.ndarray
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Converte um vetor de parâmetros em arquitetura e seus parâmetros.

        Returns:
            Tuple[str, Dict[str, Any]]: (nome_da_arquitetura, parâmetros)
        """
        return self._get_architecture_from_vector(x)

    def _generate_initial_candidates(
        self,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, Dict[str, float]]]]:
        """
        Fase 1a: Gera candidatos iniciais usando AFSA para diversificar 
        os movimentos do enxame de partículas.
        
        O AFSA explora o espaço de busca de forma inteligente usando comportamentos
        de peixes artificiais (cluster, forrageamento, movimento aleatório) para
        criar uma população inicial diversificada que será usada pelo PSO.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.ndarray, Dict[str, float]]]]: (candidatos, lista de tuplas (candidato, métricas))
        """
        # Inicializa o AFSA com os parâmetros corretos
        afsa = AFSA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            visual=0.5,  # Campo de visão para explorar o espaço
            step=0.1,  # Tamanho do passo para movimentação
            try_times=5,  # Número de tentativas para forrageamento
            max_iter=50,  # Número de iterações para otimização
            lower_bound=0.0,
            upper_bound=1.0,
        )

        # Define a função de fitness para o AFSA que incentiva a diversidade
        def afsa_fitness(x):
            # Converte os valores normalizados para parâmetros reais
            architecture_name, params = self._convert_to_architecture_params(x)

            # Calcula a diversidade baseada na variação dos parâmetros
            diversity_score = 0

            # Para a escolha da arquitetura (primeira dimensão)
            arch_diversity = (
                abs(x[0] - 0.5) * 2
            )  # Incentiva diversidade na escolha da arquitetura
            diversity_score += arch_diversity

            # Para cada parâmetro, calcula sua contribuição para a diversidade
            param_vector = x[1:]  # Pula o índice da arquitetura
            for i, (param_name, (min_val, max_val)) in enumerate(
                self.param_bounds.items()
            ):
                if i < len(param_vector):
                    normalized_value = param_vector[i]

                    # Incentiva exploração de todo o espaço de busca
                    # Valores próximos aos extremos (0 ou 1) recebem pontuação maior
                    edge_bonus = min(normalized_value, 1 - normalized_value) * 2
                    diversity_score += (
                        1 - edge_bonus
                    )  # Inverte para dar mais pontos aos extremos

                    # Adiciona variação baseada no tipo de parâmetro
                    if param_name in ["min_channels", "max_channels", "num_layers"]:
                        # Para parâmetros estruturais, incentiva mais variação
                        diversity_score += abs(normalized_value - 0.5) * 2
                    elif param_name == "dropout_rate":
                        # Para dropout, incentiva valores baixos a médios
                        diversity_score += (1 - normalized_value) * 0.5

            # Penaliza soluções muito similares na população atual
            similarity_penalty = 0
            if hasattr(afsa, "population") and len(afsa.population) > 1:
                for other_x in afsa.population:
                    if not np.array_equal(x, other_x):
                        distance = np.linalg.norm(x - other_x)
                        if distance < 0.3:  # Se muito próximos
                            similarity_penalty += (0.3 - distance) * 5

            # Score final: maximiza diversidade e minimiza similaridade
            final_score = diversity_score - similarity_penalty

            return final_score

        self._print_step("Configurando função de fitness do AFSA (baseada em diversidade)")
        afsa.fitness_function = afsa_fitness

        # Executa o AFSA para gerar candidatos
        self._print_step(f"Executando AFSA por {self.afsa_params['max_iter']} iterações")
        candidates = afsa.optimize()

        # Exibe resumo da diversidade dos candidatos gerados
        print(f"\n✅ AFSA concluído!")
        print(f"   • {len(candidates)} candidatos gerados com diversidade de arquiteturas")
        architectures_used = set()
        for candidate in candidates:
            architecture_name, _ = self._convert_to_architecture_params(candidate)
            architectures_used.add(architecture_name)
        print(f"  • Arquiteturas exploradas: {list(architectures_used)}")
        print(f"  • Parâmetros otimizados: {len(self.param_bounds)} parâmetros")

        # Realiza o warm-up dos candidatos para obter suas métricas
        self._print_section("WARM-UP: Treinando e avaliando candidatos AFSA")
        candidates_metrics = []
        for i, candidate in enumerate(tqdm(candidates, desc="Warm-up"), 1):
            print(f"\n🔄 Avaliando candidato {i}/{len(candidates)}")
            metrics = self._warm_up_candidate(candidate)
            candidates_metrics.append((candidate, metrics))
            
            # Mostra detalhes do candidato
            arch_name, arch_params = self._convert_to_architecture_params(candidate)
            oace_score = self._calculate_oace_score(metrics)
            self._print_candidate_details(candidate, metrics, arch_name, arch_params, oace_score, i)

        return candidates, candidates_metrics

    def _warm_up_candidate(self, candidate_vector: np.ndarray) -> Dict[str, float]:
        """
        Realiza o warm-up de um candidato e retorna suas métricas.
        Implementa cache para evitar re-avaliação de candidatos idênticos.

        Args:
            candidate_vector (np.ndarray): Vetor completo do candidato.

        Returns:
            Dict[str, float]: Métricas do candidato após o warm-up.
        """
        # Cria uma chave única para o candidato baseada nos valores arredondados
        # Isso evita problemas de precisão de ponto flutuante
        candidate_key = tuple(np.round(candidate_vector, decimals=4))
        
        # Verifica se já avaliamos este candidato
        if candidate_key in self.candidates_cache:
            self.cache_hits += 1
            print(f"🎯 Cache HIT! Candidato já avaliado (total hits: {self.cache_hits})")
            return self.candidates_cache[candidate_key]
        
        self.cache_misses += 1
        
        # Extrai arquitetura e parâmetros do vetor
        architecture_name, architecture_params = self._convert_to_architecture_params(
            candidate_vector
        )

        print(f"   🏗️  Arquitetura: {architecture_name}")
        print(f"   ⚙️  Parâmetros: {architecture_params}")

        # Obtém informações da arquitetura
        architecture_info = self.all_architectures[architecture_name]
        params_class = type(architecture_info["params"])

        # Cria uma instância dos parâmetros da arquitetura
        params = params_class(**architecture_params)

        # Realiza o warm-up usando a função do loader
        print(f"   🔥 Iniciando treinamento...")
        test_metrics = architecture_info["warm_up"](
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            classes=self.classes,
            num_epochs=5,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            params=params,
        )

        # Salva no cache
        self.candidates_cache[candidate_key] = test_metrics
        
        print(f"   📊 Métricas obtidas: {test_metrics}")
        print(f"   💾 Resultado salvo no cache. Total cached: {len(self.candidates_cache)}")

        return test_metrics

    def _calculate_metrics_ranges(
        self, candidates_metrics: List[Tuple[np.ndarray, Dict[str, float]]]
    ) -> None:
        """
        Calcula os limites (min/max) para cada métrica baseado nos candidatos e suas métricas já calculadas.

        Args:
            candidates_metrics (List[Tuple[np.ndarray, Dict[str, float]]]): Lista de tuplas (candidato, métricas).
        """
        print("Calculando limites das métricas...")

        # Extrai apenas as métricas da lista
        all_metrics = [metrics for _, metrics in candidates_metrics]

        # Salva no histórico
        self.metrics_history.extend(candidates_metrics)

        # Calcula os limites para cada métrica (excluindo loss por comportamento inverso)
        assertiveness_ranges = {
            "top1_acc": {
                "min": min(m["top1_acc"] for m in all_metrics),
                "max": max(m["top1_acc"] for m in all_metrics),
            },
            "top5_acc": {
                "min": min(m["top5_acc"] for m in all_metrics),
                "max": max(m["top5_acc"] for m in all_metrics),
            },
            "precision_macro": {
                "min": min(m["precision_macro"] for m in all_metrics),
                "max": max(m["precision_macro"] for m in all_metrics),
            },
            "recall_macro": {
                "min": min(m["recall_macro"] for m in all_metrics),
                "max": max(m["recall_macro"] for m in all_metrics),
            },
            "f1_macro": {
                "min": min(m["f1_macro"] for m in all_metrics),
                "max": max(m["f1_macro"] for m in all_metrics),
            },
        }

        cost_ranges = {
            "total_params": {
                "min": min(m["total_params"] for m in all_metrics),
                "max": max(m["total_params"] for m in all_metrics),
            },
            "avg_inference_time": {
                "min": min(m["avg_inference_time"] for m in all_metrics),
                "max": max(m["avg_inference_time"] for m in all_metrics),
            },
            "memory_used_mb": {
                "min": min(m["memory_used_mb"] for m in all_metrics),
                "max": max(m["memory_used_mb"] for m in all_metrics),
            },
            "gflops": {
                "min": min(m["gflops"] for m in all_metrics),
                "max": max(m["gflops"] for m in all_metrics),
            },
        }

        self.metrics_ranges = {
            "assertiveness": assertiveness_ranges,
            "cost": cost_ranges,
        }

    def fitness_function(self, x: np.ndarray) -> float:
        """
        Função de fitness que avalia um candidato usando o score OACE.
        
        Args:
            x (np.ndarray): Vetor de parâmetros normalizados
            
        Returns:
            float: Score OACE (entre 0 e 1)
        """
        # Converte o vetor para string para usar como chave do cache
        cache_key = str(x.tolist())
        
        # Verifica se já foi avaliado
        if cache_key in self.candidates_cache:
            self.cache_hits += 1
            print(f"🎯 Cache HIT! Candidato já avaliado (total hits: {self.cache_hits})")
            return self.candidates_cache[cache_key]
        
        # Se não está no cache, avalia o candidato
        self.cache_misses += 1
        print(f"🆕 Novo candidato avaliado")
        
        # Treina e avalia o candidato
        metrics = self._warm_up_candidate(x)
        
        # Calcula o score OACE
        score = self._calculate_oace_score(metrics)
        
        # Armazena no cache
        self.candidates_cache[cache_key] = score
        
        print(f"   🎯 Score OACE calculado: {score:.6f}")
        
        # Garante que o score está dentro do range válido
        if score > 1.0:
            print(f"   ⚠️  AVISO: Score OACE inválido ({score:.6f}) > 1.0. Corrigindo...")
            score = 1.0
        elif score < 0.0:
            print(f"   ⚠️  AVISO: Score OACE inválido ({score:.6f}) < 0.0. Corrigindo...")
            score = 0.0
        
        return score

    def initialize_components(self):
        """
        Inicializa os componentes do algoritmo híbrido.
        """
        # Inicializa o AFSA
        self.afsa = AFSA(
            population_size=self.population_size,
            n_dim=self.n_dim,
            visual=self.afsa_params["visual"],
            step=self.afsa_params["step"],
            try_times=self.afsa_params["try_times"],
            max_iter=self.afsa_params["max_iter"],
            lower_bound=0.0,  # Normalizado para [0,1]
            upper_bound=1.0,
        )
        
        # Inicializa o PSO
        self.pso = PSO(
            population_size=self.population_size,
            n_dim=self.n_dim,
            max_iter=self.max_iter,
            lower_bound=0.0,
            upper_bound=1.0,
            afsa_params=self.afsa_params,
            pso_options=self.pso_params,
        )
        
        # GA removido

    def optimize(self):
        """
        Executa o processo de otimização híbrida AFSA-PSO seguindo o fluxo correto:
        
        AFSA-PSO (Otimização Inicial)
        - AFSA diversifica movimentos do enxame de partículas
        - PSO com AFSA gera "soluções de otimização inicial"
        
        Returns:
            tuple: (melhor arquitetura encontrada, melhor valor de fitness)
        """
        self._print_header("INICIANDO OTIMIZAÇÃO HÍBRIDA AFSA-PSO")
        self._print_configuration()
        
        try:
            # Passo 1: Inicializa os componentes
            self._print_step("Inicializando componentes do algoritmo híbrido")
            self.initialize_components()


            # FASE 1: OTIMIZAÇÃO INICIAL COM AFSA-PSO
            self._print_section("OTIMIZAÇÃO COM AFSA-PSO")
            
            # Gera população inicial usando AFSA para diversificar movimentos
            self._print_step("Gerando população inicial diversificada com AFSA", 
                           f"Tamanho: {self.population_size}, Iterações: {self.afsa_params['max_iter']}")
            initial_population, candidates_metrics = self._generate_initial_candidates()
            
            # Calcula limites das méctricas usando as métricas já calculadas
            self._print_step("Calculando limites das métricas para normalização OACE")
            self._calculate_metrics_ranges(candidates_metrics)
            
            # Executa PSO com população inicial do AFSA
            self._print_step("Executando PSO com população inicial diversificada", 
                           f"Iterações: {self.max_iter}, Parâmetros: {self.pso_params}")
            phase1_solutions = self._execute_afsa_pso_phase(initial_population, candidates_metrics)
            
            # Seleciona melhor solução da Fase 1 e finaliza
            best_idx = np.argmax([self.fitness_function(x) for x in phase1_solutions])
            best_position = phase1_solutions[best_idx]
            best_fitness = self.fitness_function(best_position)
            best_arch, best_params = self._convert_to_architecture_params(best_position)
            self._print_phase_summary("AFSA-PSO", best_fitness, best_arch, best_params)

            self.best_solution = best_position
            self.best_fitness = best_fitness

            best_architecture_name, best_architecture_params = (
                self._convert_to_architecture_params(best_position)
            )

            # Obtém as métricas finais
            self._print_step("Avaliando solução final para métricas completas")
            final_metrics = self._warm_up_candidate(best_position)


            # Imprime resultados finais formatados
            self._print_final_results(best_architecture_name, best_architecture_params, 
                                    best_fitness, final_metrics)

            return best_architecture_name, best_architecture_params, best_fitness
            
        except Exception as e:
            print(f"\n❌ Erro durante a otimização: {str(e)}")
            raise

    def _execute_afsa_pso_phase(self, initial_population, candidates_metrics):
        """
        Executa a Fase 1: AFSA-PSO (Otimização Inicial)
        """
        self._print_section("AFSA-PSO: Inicializando PSO com soluções da Fase 1")
        
        self._print_step("Calculando fitness dos candidatos iniciais com OACE")
        initial_fitness = []
        for i, (candidate, metrics) in enumerate(candidates_metrics, 1):
            print(f"\n🔄 Avaliando candidato {i}/{len(candidates_metrics)}")
            score = self._calculate_oace_score(metrics)
            initial_fitness.append(score)
            print(f"   🎯 Score OACE: {score:.6f}")

        initial_fitness = np.array(initial_fitness)
        best_idx = np.argmax(initial_fitness)
        
        # Mostra resumo da população inicial
        self._print_population_summary(initial_population, initial_fitness, "AFSA-PSO Inicial")
        
        print(f"\n🏆 Melhor candidato inicial:")
        print(f"   • Índice: {best_idx}")
        print(f"   • Score OACE: {initial_fitness[best_idx]:.6f}")
        arch_name, arch_params = self._convert_to_architecture_params(initial_population[best_idx])
        print(f"   • Arquitetura: {arch_name}")
        print(f"   • Parâmetros: {arch_params}")

        def pso_fitness_function(x):
            if x.ndim == 1:
                return -self.fitness_function(x)
            else:
                scores = []
                for xi in x:
                    score = self.fitness_function(xi)
                    scores.append(score)
                return -np.array(scores)

        self.pso.fitness_function = pso_fitness_function
        
        # Inicializa completamente o enxame do PSO com a população do AFSA
        self._print_step("Inicializando enxame PSO com população do AFSA")
        self.pso.initialize_swarm_with_population(initial_population)
        
        self._print_step("PSO explorando espaço de busca e gerando novos candidatos", 
                        f"Iterações: {self.max_iter}")
        best_pos, best_cost = self.pso.optimize(metrics_function=self._warm_up_candidate)
        
        # Converte o custo interno (minimização) para score OACE (maximização)
        oace_score = -float(best_cost) if best_cost is not None else 0.0
        print(f"\n🏆 PSO Concluído!")
        print(f"   • Melhor posição: {best_pos}")
        print(f"   • Score OACE: {oace_score:.6f}")

        final_population = self.pso.optimizer.swarm.position
        
        # Garante que o melhor global (best_pos) também seja avaliado
        all_candidates = np.vstack([final_population, best_pos.reshape(1, -1)])
        self._print_step(f"Avaliando {len(all_candidates)} soluções finais do PSO (incluindo best_pos)")
        
        final_fitness = []
        for i, pos in enumerate(all_candidates):
            print(f"   🔄 Avaliando solução {i+1}/{len(all_candidates)}")
            fitness = self.fitness_function(pos)
            final_fitness.append(fitness)
            print(f"      🎯 Score OACE: {fitness:.6f}")
        
        final_fitness = np.array(final_fitness)
        
        # Seleciona os melhores (max OACE)
        best_indices = np.argsort(final_fitness)[-self.population_size:]
        phase1_solutions = all_candidates[best_indices]

        self._print_population_summary(phase1_solutions, final_fitness[best_indices], "AFSA-PSO Final")
        
        print(f"\n✅ Fase AFSA-PSO Concluída!")
        print(f"   • Melhor score da Fase 1: {np.max(final_fitness):.6f}")
        print(f"   • {len(phase1_solutions)} soluções selecionadas para Fase 2")
        
        return phase1_solutions

    # Método de GA removido

    def _calculate_oace_score(self, metrics):
        """
        Calcula o score OACE para um conjunto de métricas.
        
        Args:
            metrics: Dicionário com as métricas
            
        Returns:
            float: Score OACE (entre 0 e 1)
        """
        print(f"   📊 Calculando score OACE...")
        
        # Usa apenas métricas positivas para assertividade (não inclui loss)
        assertiveness_weights = {
            "top1_acc": 0.4,        # Peso maior para acurácia principal
            "top5_acc": 0.15,       
            "precision_macro": 0.25,
            "recall_macro": 0.15,
            "f1_macro": 0.05,       # Peso menor pois f1 é derivado de precision/recall
        }
        cost_weights = {
            "total_params": 0.25,
            "avg_inference_time": 0.25,
            "memory_used_mb": 0.25,
            "gflops": 0.25,
        }

        # Atualiza os limites dinamicamente para incluir novos valores
        self._update_metrics_ranges(metrics)
        
        print(f"      • Limites assertividade: {self.metrics_ranges['assertiveness']}")
        print(f"      • Limites custo: {self.metrics_ranges['cost']}")
        print(f"      • Lambda (trade-off): {self.lambda_param}")

        # Calcula o score OACE usando os limites atualizados
        score = calculate_oace_score(
            assertiveness_metrics={
                "top1_acc": metrics["top1_acc"], 
                "top5_acc": metrics["top5_acc"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            },
            cost_metrics={
                "total_params": metrics["total_params"],
                "avg_inference_time": metrics["avg_inference_time"],
                "memory_used_mb": metrics["memory_used_mb"],
                "gflops": metrics["gflops"],
            },
            lambda_param=self.lambda_param,
            assertiveness_weights=assertiveness_weights,
            cost_weights=cost_weights,
            assertiveness_min_max=self.metrics_ranges["assertiveness"],
            cost_min_max=self.metrics_ranges["cost"],
        )
        
        print(f"      • Score OACE calculado: {score:.6f}")
        
        # Validação: Garante que o score está no range correto
        if not (0.0 <= score <= 1.0):
            print(f"   ⚠️  AVISO: Score OACE fora do range [0,1]: {score:.6f}")
            print(f"   Métricas de assertividade:")
            for key in ["top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro"]:
                if key in metrics:
                    print(f"     {key}: {metrics[key]:.4f}")
            print(f"   Métricas de custo:")
            for key in ["total_params", "avg_inference_time", "memory_used_mb", "gflops"]:
                if key in metrics:
                    print(f"     {key}: {metrics[key]:.4f}")
            print(f"   Limites de assertividade: {self.metrics_ranges['assertiveness']}")
            print(f"   Limites de custo: {self.metrics_ranges['cost']}")
            # Clipa o valor para o range válido
            score = max(0.0, min(1.0, score))
            print(f"   Score corrigido: {score:.6f}")
        
        return score

    def _update_metrics_ranges(self, new_metrics):
        """
        Atualiza os limites min/max das métricas dinamicamente para incluir novos valores.
        Isso evita que candidatos fiquem fora do range e gerem scores negativos.
        
        Args:
            new_metrics: Dicionário com novas métricas a serem incluídas nos limites
        """
        if self.metrics_ranges is None:
            print("   🔧 Inicializando limites das métricas com valores padrão...")
            self.metrics_ranges = {
                "assertiveness": {
                    "top1_acc": {"min": new_metrics.get("top1_acc", 0.0), "max": new_metrics.get("top1_acc", 1.0)},
                    "top5_acc": {"min": new_metrics.get("top5_acc", 0.0), "max": new_metrics.get("top5_acc", 1.0)},
                    "precision_macro": {"min": new_metrics.get("precision_macro", 0.0), "max": new_metrics.get("precision_macro", 1.0)},
                    "recall_macro": {"min": new_metrics.get("recall_macro", 0.0), "max": new_metrics.get("recall_macro", 1.0)},
                    "f1_macro": {"min": new_metrics.get("f1_macro", 0.0), "max": new_metrics.get("f1_macro", 1.0)},
                },
                "cost": {
                    "total_params": {"min": new_metrics.get("total_params", 0), "max": new_metrics.get("total_params", 1000000)},
                    "avg_inference_time": {"min": new_metrics.get("avg_inference_time", 0.0), "max": new_metrics.get("avg_inference_time", 1.0)},
                    "memory_used_mb": {"min": new_metrics.get("memory_used_mb", 0.0), "max": new_metrics.get("memory_used_mb", 1000.0)},
                    "gflops": {"min": new_metrics.get("gflops", 0.0), "max": new_metrics.get("gflops", 100.0)},
                }
            }
            print(f"   ✅ Limites inicializados com valores atuais")
            return
            
        # Atualiza limites de assertividade
        assertiveness_metrics = ["top1_acc", "top5_acc", "precision_macro", "recall_macro", "f1_macro"]
        for metric in assertiveness_metrics:
            if metric in new_metrics and metric in self.metrics_ranges["assertiveness"]:
                current_min = self.metrics_ranges["assertiveness"][metric]["min"]
                current_max = self.metrics_ranges["assertiveness"][metric]["max"]
                new_value = new_metrics[metric]
                
                # Atualiza min/max se necessário
                old_min, old_max = current_min, current_max
                self.metrics_ranges["assertiveness"][metric]["min"] = min(current_min, new_value)
                self.metrics_ranges["assertiveness"][metric]["max"] = max(current_max, new_value)
                
                # Log da atualização se houve mudança
                if old_min != self.metrics_ranges["assertiveness"][metric]["min"] or old_max != self.metrics_ranges["assertiveness"][metric]["max"]:
                    print(f"   📊 Atualizado limite {metric}: [{old_min:.4f}, {old_max:.4f}] → [{self.metrics_ranges['assertiveness'][metric]['min']:.4f}, {self.metrics_ranges['assertiveness'][metric]['max']:.4f}]")
        
        
        # Atualiza limites de custo
        cost_metrics = ["total_params", "avg_inference_time", "memory_used_mb", "gflops"]
        for metric in cost_metrics:
            if metric in new_metrics and metric in self.metrics_ranges["cost"]:
                current_min = self.metrics_ranges["cost"][metric]["min"]
                current_max = self.metrics_ranges["cost"][metric]["max"]
                new_value = new_metrics[metric]
                
                # Atualiza min/max se necessário
                old_min, old_max = current_min, current_max
                self.metrics_ranges["cost"][metric]["min"] = min(current_min, new_value)
                self.metrics_ranges["cost"][metric]["max"] = max(current_max, new_value)
                
                # Log da atualização se houve mudança
                if old_min != self.metrics_ranges["cost"][metric]["min"] or old_max != self.metrics_ranges["cost"][metric]["max"]:
                    print(f"   📊 Atualizado limite {metric}: [{old_min:.4f}, {old_max:.4f}] → [{self.metrics_ranges['cost'][metric]['min']:.4f}, {self.metrics_ranges['cost'][metric]['max']:.4f}]")

# Exemplo de uso:
if __name__ == "__main__":
    
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    
    # Carregar os data loaders
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()

    # Criar instância do otimizador híbrido (com parâmetros reduzidos para teste)
    optimizer = AFSAPSO(
        population_size=15,
        max_iter=15,  
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        lambda_param=0.5,
        afsa_params={'visual': 150, 'step': 5, 'try_times': 5, 'max_iter': 15},  # Reduzido
        architectures_to_optimize=['CNN', 'ResNet', 'EfficientNet', 'MobileNet']  # ['CNN', 'ResNet', 'EfficientNet', 'MobileNet']
    )
    # Executa a otimização
    best_architecture, best_params, best_fitness = optimizer.optimize()
    results = best_architecture, best_params, best_fitness
    
    print(f"\nMelhor arquitetura encontrada: {best_architecture}")
    print(f"Parâmetros da melhor arquitetura: {best_params}")
    print(f"Melhor valor de fitness (OACE): {best_fitness}")
    
    print("results: ", results)

