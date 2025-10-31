# Análise: Integração de MCDMs (OACE, TOPSIS, VIKOR) no PSO

## Situação Atual

### Estrutura do PSO

1. **PSO (`pso.py`)**:
   - Função `fitness_function` recebe **batch de partículas** (`x: np.ndarray` shape `(n_particles, n_dim)`)
   - Retorna array de fitness `(n_particles,)` - um valor por partícula
   - Avaliação é **vetorizada/paralela** - todas as partículas são avaliadas simultaneamente
   - Cada iteração: todas as partículas são avaliadas de uma vez

2. **AFSA-PSO (`afsa_pso.py`)**:
   - Função `fitness_function` recebe **uma partícula** (`x: np.ndarray` shape `(n_dim,)`)
   - Retorna **float** (um valor)
   - Usa OACE iterativamente - avalia um candidato por vez
   - Treina modelo, coleta métricas, calcula OACE

### Características dos MCDMs

| Método | Tipo de Avaliação | Precisa de Contexto? | Normalização |
|--------|-------------------|---------------------|--------------|
| **OACE** | **Individual** | ❌ Não | Min-max dinâmico (atualiza limites conforme encontra novos valores) |
| **TOPSIS** | **Contextual** | ✅ Sim | Precisa de TODAS as alternativas para calcular ideais (normalização vector ou min-max do conjunto) |
| **VIKOR** | **Contextual** | ✅ Sim | Precisa de TODAS as alternativas para calcular S, R, Q (normalização min-max do conjunto) |

---

## Desafios Técnicos

### 1. TOPSIS e VIKOR são **Contextuais**

**Problema Fundamental**:
- TOPSIS calcula soluções ideais positivas/negativas baseadas no **conjunto completo** de alternativas
- VIKOR calcula S, R, Q baseados nos **valores máximo/mínimo** do conjunto completo
- O ranking de uma alternativa **depende** de todas as outras alternativas na mesma geração

**Exemplo**:
```
Geração 1: [Modelo A: accuracy=0.85, Modelo B: accuracy=0.80]
  - TOPSIS: Modelo A é ideal (melhor accuracy)

Geração 2: [Modelo A: accuracy=0.85, Modelo B: accuracy=0.80, Modelo C: accuracy=0.90]
  - TOPSIS: Modelo C agora é ideal (melhor accuracy)
  - Ranking do Modelo A muda mesmo que suas métricas não mudem!
```

### 2. OACE é **Iterativo**

**Vantagem**:
- Avalia cada alternativa **independentemente**
- Limites min/max são atualizados dinamicamente conforme novos valores são encontrados
- Score de uma alternativa não depende de outras alternativas na mesma geração

**Como funciona**:
```python
# Avalia Modelo A
oace.evaluate([metrics_A])  # Score baseado em limites históricos

# Avalia Modelo B
oace.evaluate([metrics_B])  # Score baseado em limites atualizados (pode incluir A)

# Limites se expandem conforme encontra novos extremos
```

---

## Soluções Possíveis

### ✅ **Solução 1: Avaliação por Geração (Recomendada)**

**Estratégia**: Avaliar todas as partículas da geração, depois aplicar MCDM contextual.

#### Fluxo:

```
ITERAÇÃO DO PSO:
1. PSO gera novas posições para todas as partículas
2. FASE DE TREINAMENTO: Treina todos os modelos em paralelo/sequencial
3. FASE DE COLETA: Coleta métricas de todos os modelos
4. FASE DE AVALIAÇÃO MCDM:
   
   Se OACE:
     - Avalia cada partícula individualmente
     - Atualiza limites min/max conforme encontra novos extremos
     - Retorna scores individuais
   
   Se TOPSIS/VIKOR:
     - Constrói matriz de decisão (n_partículas × n_critérios)
     - Aplica TOPSIS/VIKOR na matriz completa
     - Retorna scores baseados no ranking da geração
5. Atualiza pbest/gbest do PSO com os scores
```

#### Vantagens:
- ✅ **Comparação justa**: Todos os métodos avaliam a mesma geração
- ✅ **TOPSIS/VIKOR funcionam corretamente**: Têm acesso ao contexto completo
- ✅ **OACE funciona normalmente**: Continua avaliando individualmente
- ✅ **Implementação clara**: Separação entre treinamento e avaliação

#### Desafios:
- ⚠️ PSO precisa ser modificado para suportar avaliação em duas fases
- ⚠️ TOPSIS/VIKOR precisam recalcular ranking a cada geração (ranking pode mudar entre gerações)

#### Estrutura Proposta:

```python
class PSOWithMCDM:
    def __init__(self, mcdm_method="OACE", ...):
        self.mcdm_method = mcdm_method  # "OACE", "TOPSIS", "VIKOR"
        self.mcdm_evaluator = self._create_mcdm_evaluator()
    
    def _create_mcdm_evaluator(self):
        if self.mcdm_method == "OACE":
            return OACE(...)
        elif self.mcdm_method == "TOPSIS":
            return TOPSIS(...)
        elif self.mcdm_method == "VIKOR":
            return VIKOR(...)
    
    def evaluate_generation(self, particles_positions):
        """
        Avalia uma geração completa de partículas.
        
        Returns:
            fitness_scores: array (n_particles,) com scores
        """
        # Fase 1: Treinar e coletar métricas
        all_metrics = []
        for pos in particles_positions:
            metrics = self._train_and_evaluate(pos)
            all_metrics.append(metrics)
        
        # Fase 2: Aplicar MCDM
        decision_matrix = self._build_decision_matrix(all_metrics)
        
        if self.mcdm_method == "OACE":
            # Avalia individualmente, atualizando limites
            scores = []
            for metrics in all_metrics:
                row = self._metrics_to_row(metrics)
                result = self.mcdm_evaluator.evaluate(row.reshape(1, -1))
                scores.append(result["scores"][0])
                # Atualiza limites históricos
                self._update_oace_limits(metrics)
            return np.array(scores)
        
        else:  # TOPSIS ou VIKOR
            # Avalia contexto completo
            result = self.mcdm_evaluator.evaluate(decision_matrix)
            return result["scores"]  # Scores já estão no contexto da geração
```

---

### ✅ **Solução 2: Wrapper Contextual para OACE**

**Estratégia**: Fazer OACE também considerar contexto quando necessário, mas manter compatibilidade com avaliação individual.

#### Fluxo:

```
ITERAÇÃO DO PSO:
1. Treina todas as partículas
2. Coleta todas as métricas
3. Aplica MCDM:
   
   Se OACE:
     - Opção A: Avalia individualmente (modo atual)
     - Opção B: Avalia com contexto (usar min/max da geração atual)
   
   Se TOPSIS/VIKOR:
     - Sempre avalia com contexto (geração completa)
```

#### Vantagens:
- ✅ Permite comparação entre OACE contextual vs individual
- ✅ Mantém flexibilidade do OACE
- ✅ Todos os métodos podem usar mesmo paradigma

#### Desafios:
- ⚠️ Adiciona complexidade ao OACE
- ⚠️ Precisamos decidir qual modo usar para comparação justa

---

### ✅ **Solução 3: Histórico Adaptativo**

**Estratégia**: Manter histórico de todas as partículas já avaliadas, usar para calcular contexto.

#### Fluxo:

```
ITERAÇÃO DO PSO:
1. Treina partículas
2. Coleta métricas
3. Adiciona ao histórico global
4. Aplica MCDM usando histórico + partículas atuais

Para TOPSIS/VIKOR:
  - Usa histórico completo (todas as gerações)
  - Ranking pode mudar conforme novas partículas são adicionadas

Para OACE:
  - Continua individual, mas limites incluem histórico
```

#### Vantagens:
- ✅ TOPSIS/VIKOR têm contexto rico (todas as partículas já avaliadas)
- ✅ Rankings mais estáveis entre gerações

#### Desafios:
- ⚠️ Ranking pode mudar retroativamente (partícula boa pode virar ruim quando novas são adicionadas)
- ⚠️ Memória cresce com número de iterações
- ⚠️ **NÃO É JUSTO** comparar com OACE que só vê geração atual

---

## Recomendações para Comparação Justa

### ✅ **Abordagem Recomendada: Avaliação por Geração**

**Princípio**: Todos os métodos avaliam **apenas a geração atual** de partículas.

#### Estrutura:

```python
class PSOMCDMWrapper:
    """
    Wrapper que permite usar diferentes MCDMs no PSO.
    Gerencia avaliação por geração para compatibilidade.
    """
    
    def __init__(self, mcdm_method="OACE", ...):
        self.mcdm_method = mcdm_method
        # ... inicialização
    
    def fitness_function(self, particles_positions):
        """
        Avalia batch de partículas usando MCDM escolhido.
        
        Args:
            particles_positions: array (n_particles, n_dim)
        
        Returns:
            fitness_scores: array (n_particles,)
        """
        # 1. Treinar e coletar métricas de todas as partículas
        all_metrics = self._train_all_particles(particles_positions)
        
        # 2. Construir matriz de decisão
        decision_matrix = self._build_decision_matrix(all_metrics)
        
        # 3. Aplicar MCDM apropriado
        if self.mcdm_method == "OACE":
            return self._evaluate_oace(all_metrics)
        elif self.mcdm_method == "TOPSIS":
            return self._evaluate_topsis(decision_matrix)
        elif self.mcdm_method == "VIKOR":
            return self._evaluate_vikor(decision_matrix)
    
    def _evaluate_oace(self, all_metrics):
        """Avalia cada partícula individualmente com OACE."""
        scores = []
        for metrics in all_metrics:
            row = self._metrics_to_row(metrics)
            # Atualiza limites históricos antes de avaliar
            self._update_oace_limits(metrics)
            # Avalia individualmente
            result = self.oace_evaluator.evaluate(
                row.reshape(1, -1),
                self.criteria_weights,
                self.criteria_types
            )
            scores.append(result["scores"][0])
        return np.array(scores)
    
    def _evaluate_topsis(self, decision_matrix):
        """Avalia geração completa com TOPSIS."""
        result = self.topsis_evaluator.evaluate(
            decision_matrix,
            self.criteria_weights,
            self.criteria_types
        )
        return result["scores"]
    
    def _evaluate_vikor(self, decision_matrix):
        """Avalia geração completa com VIKOR."""
        result = self.vikor_evaluator.evaluate(
            decision_matrix,
            self.criteria_weights,
            self.criteria_types
        )
        # VIKOR retorna Q (menor = melhor), precisa inverter para PSO (menor = melhor)
        return 1.0 - result["scores"]  # Inverte: maior Q = menor fitness
```

### Pontos Importantes:

1. **Normalização Consistente**:
   - **TOPSIS/VIKOR**: Normalizam baseado na **geração atual** (min/max das partículas da geração)
   - **OACE**: Normaliza baseado em **histórico acumulado** (min/max de todas as partículas já vistas)
   - **⚠️ Diferença**: Isso cria uma pequena assimetria, mas é inerente aos métodos

2. **Sinal de Fitness**:
   - PSO busca **minimizar** fitness
   - TOPSIS: Score maior = melhor → precisa **inverter** (`fitness = 1 - topsis_score`)
   - VIKOR: Q menor = melhor → precisa **inverter** (`fitness = 1 - q_score` ou usar Q diretamente)
   - OACE: Score maior = melhor → precisa **inverter** (`fitness = 1 - oace_score`)

3. **Cache e Reavaliação**:
   - Partículas podem ser reavaliadas entre iterações
   - **OACE**: Score pode mudar se limites mudarem
   - **TOPSIS/VIKOR**: Score pode mudar se contexto da geração mudar
   - ⚠️ Precisa gerenciar cache cuidadosamente

---

## Comparação Justa entre MCDMs

### Métricas de Comparação:

1. **Convergência**:
   - Número de iterações para convergir
   - Valor final do melhor fitness
   - Qualidade da melhor arquitetura encontrada

2. **Diversidade**:
   - Variância das partículas ao longo das iterações
   - Exploração vs Exploração

3. **Eficiência**:
   - Tempo de avaliação por geração
   - Número de avaliações necessárias

4. **Robustez**:
   - Consistência dos resultados entre execuções
   - Sensibilidade a hiperparâmetros

### Experimento Proposto:

```
Para cada MCDM (OACE, TOPSIS, VIKOR):
  1. Executar PSO com mesma semente (mesma inicialização)
  2. Usar mesmos hiperparâmetros do PSO
  3. Usar mesma função de treinamento/avaliação
  4. Comparar:
     - Trajetória do melhor fitness ao longo das iterações
     - Arquitetura final encontrada
     - Tempo de execução
     - Diversidade da população
```

---

## Conclusão

### ✅ **É VIÁVEL** integrar os três MCDMs no PSO

**Estrutura Recomendada**:

1. **Wrapper `PSOMCDMWrapper`**:
   - Gerencia treinamento de todas as partículas
   - Coleciona métricas
   - Aplica MCDM apropriado (OACE individual, TOPSIS/VIKOR contextual)
   - Retorna scores normalizados para PSO

2. **Modificações Necessárias**:
   - Modificar `pso.py` para aceitar função de fitness que trabalha com geração completa
   - Ou criar wrapper que adapta avaliação individual → batch
   - Gerenciar normalização e inversão de sinais adequadamente

3. **Comparação Justa**:
   - Todos avaliam mesma geração
   - Todos usam mesmos dados de treinamento
   - Diferenças são apenas metodológicas (OACE histórico vs TOPSIS/VIKOR contextual)

### Próximos Passos Sugeridos:

1. Implementar `PSOMCDMWrapper` com suporte aos três métodos
2. Adicionar flag `mcdm_method` no PSO
3. Criar script de experimento comparativo
4. Executar múltiplas rodadas e analisar estatisticamente

**A implementação é viável e permitirá comparação justa entre os MCDMs!**

## Passos para estruturar a solução

### Fase 1: Preparação e estruturação base

1. Criar classe `PSOMCDMWrapper` para gerenciar a integração dos MCDMs com o PSO.

2. Definir interface comum para os três MCDMs:
   - Método para identificar tipo de avaliação (individual vs contextual)
   - Método para conversão de métricas em formato de matriz de decisão

3. Criar função auxiliar para construção da matriz de decisão:
   - Recebe lista de métricas de todas as partículas
   - Retorna matriz (n_partículas × n_critérios)
   - Define ordem e mapeamento dos critérios

4. Criar função para normalizar sinais de fitness:
   - PSO minimiza, MCDMs maximizam
   - Converter: `fitness = 1.0 - mcdm_score` (ou similar)

### Fase 2: Implementação do wrapper

5. Implementar método `_train_all_particles`:
   - Recebe posições das partículas
   - Treina cada uma (usar função de treinamento existente)
   - Retorna lista de métricas

6. Implementar método `_build_decision_matrix`:
   - Converte lista de métricas em matriz NumPy
   - Ordem consistente: accuracy, top5, precision, recall, f1, params, time, mem, gflops

7. Implementar método `_evaluate_oace`:
   - Avalia cada partícula individualmente
   - Atualiza limites históricos (min/max) conforme avalia
   - Retorna array de scores

8. Implementar método `_evaluate_topsis`:
   - Recebe matriz de decisão completa
   - Aplica TOPSIS na geração toda
   - Inverte scores para fitness (menor = melhor)

9. Implementar método `_evaluate_vikor`:
   - Recebe matriz de decisão completa
   - Aplica VIKOR na geração toda
   - Inverte Q para fitness (menor = melhor)

10. Implementar método `fitness_function` principal:
    - Recebe batch de posições de partículas
    - Chama `_train_all_particles`
    - Chama `_build_decision_matrix`
    - Chama método de avaliação conforme `mcdm_method`
    - Retorna array de fitness scores

### Fase 3: Integração com PSO existente

11. Modificar `pso.py` para aceitar `mcdm_method` como parâmetro.

12. Adaptar `fitness_function` do PSO para usar o wrapper:
    - Instanciar `PSOMCDMWrapper` no `__init__`
    - Passar função de treinamento para o wrapper
    - Usar wrapper.fitness_function no lugar da função atual

13. Gerenciar limites históricos para OACE:
    - Manter estado global de min/max por critério
    - Atualizar a cada avaliação individual
    - Inicializar com valores razoáveis

### Fase 4: Configuração e parâmetros

14. Definir estrutura de configuração:
    - Pesos dos critérios
    - Tipos dos critérios (benefit/cost)
    - Parâmetros específicos (lambda para OACE, v para VIKOR, etc.)

15. Criar função de inicialização:
    - Configura OACE, TOPSIS, VIKOR com mesmos pesos/tipos
    - Garante consistência na comparação

### Fase 5: Validação e testes

16. Criar teste unitário:
    - Verifica que os três métodos recebem as mesmas partículas
    - Valida formato de saída (array de fitness)

17. Criar teste de comparação:
    - Executa os três métodos com mesma semente
    - Compara resultados para validar diferenças metodológicas

18. Adicionar logging/comparação:
    - Registra qual método está sendo usado
    - Registra métricas de cada geração para análise posterior

### Fase 6: Experimentação comparativa

19. Criar script de experimento:
    - Executa PSO com cada MCDM separadamente
    - Salva trajetórias de fitness, melhores arquiteturas, tempos

20. Implementar análise comparativa:
    - Compara convergência, qualidade final, eficiência
    - Gera gráficos e relatórios estatísticos

### Ordem de implementação sugerida

- Passos 1-4: Estrutura base
- Passos 5-10: Implementação do wrapper
- Passos 11-13: Integração com PSO
- Passos 14-15: Configuração
- Passos 16-18: Validação
- Passos 19-20: Experimentação

Esses passos estruturam a solução de avaliação por geração permitindo comparação justa entre os três MCDMs.