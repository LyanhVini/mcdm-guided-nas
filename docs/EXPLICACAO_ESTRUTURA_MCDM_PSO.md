# Explicação Detalhada: Estrutura MCDM-PSO e Conexões

## 📋 O Que Foi Implementado

### **Passos 1 e 2: Estrutura Base** ✅
- **`PSOMCDMWrapper`**: Classe principal que gerencia a integração
- **Interface de tipos**: Mapeamento de métodos para tipo de avaliação

### **Passos 3 e 4: Funções Auxiliares** ✅
- **`build_decision_matrix()`**: Constrói matriz de decisão a partir de métricas
- **`mcdm_to_fitness()`**: Converte scores MCDM para fitness PSO

---

## 🔧 Detalhamento das Funções Auxiliares

### 1. `build_decision_matrix()` - Construção de Matriz

**Localização**: `src/mcdm/utils.py`

**O que faz:**
```python
# Recebe: Lista de dicionários com métricas
all_metrics = [
    {"top1_acc": 0.85, "total_params": 3e6, ...},  # Partícula 1
    {"top1_acc": 0.80, "total_params": 5e6, ...},  # Partícula 2
    ...
]

# Retorna: Matriz NumPy estruturada
decision_matrix = [
    [0.85, 0.92, ..., 3e6, ...],  # Linha 1 = Partícula 1
    [0.80, 0.90, ..., 5e6, ...],  # Linha 2 = Partícula 2
    ...
]
# Shape: (n_partículas, 9 critérios)
```

**Por que é importante:**
- **Padronização**: Garante ordem consistente dos critérios (sempre: top1, top5, precision, recall, f1, params, time, mem, gflops)
- **Compatibilidade**: TOPSIS e VIKOR precisam de matriz NumPy estruturada
- **Reutilização**: Uma função única usada por todos os MCDMs

**Características:**
- Trata métricas faltantes (preenche com 0.0)
- Ordem fixa e conhecida (DEFAULT_CRITERIA_ORDER)
- Funciona com qualquer número de partículas

---

### 2. `mcdm_to_fitness()` - Conversão de Scores

**Localização**: `src/mcdm/utils.py`

**O que faz:**
```python
# MCDMs retornam scores onde MAIOR = MELHOR
mcdm_scores = [0.9, 0.7, 0.8, 0.6]  # 0.9 é o melhor

# PSO precisa de fitness onde MENOR = MELHOR
fitness = mcdm_to_fitness(mcdm_scores, invert=True)
# Resultado: [0.1, 0.3, 0.2, 0.4]  # 0.1 é o melhor (invertido)

# O ranking é preservado:
# Melhor MCDM (0.9) → Melhor Fitness (0.1)
# Mesmo índice, valores invertidos
```

**Por que é importante:**
- **Incompatibilidade de sinais**: MCDMs maximizam, PSO minimiza
- **Ranking preservado**: O melhor continua sendo o melhor, apenas invertido
- **Flexibilidade**: Opção `invert=False` para MCDMs que já minimizam

**Características:**
- Inversão simples: `fitness = 1.0 - score`
- Preserva ordem relativa (ranking não muda)
- Funciona com arrays NumPy de qualquer tamanho

---

## 🔄 Como Tudo Se Conecta: Fluxo Completo

### **Arquitetura do Sistema:**

```
┌─────────────────────────────────────────────────────────────┐
│                    PSO (optimizers/pso.py)                   │
│  - Gerencia enxame de partículas                            │
│  - Atualiza posições e velocidades                          │
│  - Precisa de fitness_function()                             │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                │ Chama fitness_function(batch)
                                ▼
┌─────────────────────────────────────────────────────────────┐
│         PSOMCDMWrapper (optimizers/pso_mcdm_wrapper.py)      │
│  - Recebe batch de partículas                                │
│  - Coordena avaliação usando MCDM                            │
│  - Retorna fitness para PSO                                  │
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────────┐      ┌───────────────────────┐
    │  Fase 1: Treinar      │      │  build_decision_matrix│
    │  - Para cada          │      │  (mcdm/utils.py)      │
    │    partícula:         │      │  - Converte dict →   │
    │    train_function()    │      │    matriz NumPy      │
    │    retorna métricas   │      └───────────────────────┘
    └───────────────────────┘                 │
                │                               │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────┐
                │  Fase 2: Matriz       │
                │  Shape: (n, 9)        │
                └───────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────────┐      ┌───────────────────────┐
    │  OACE (individual)    │      │  TOPSIS/VIKOR         │
    │  - Avalia uma por vez │      │  (contextual)         │
    │  - Atualiza limites   │      │  - Avalia matriz      │
    │  - Score independente │      │    completa           │
    └───────────────────────┘      │  - Ranking relativo  │
                │                   └───────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────┐
                │  mcdm_to_fitness()     │
                │  (mcdm/utils.py)       │
                │  - Inverte scores      │
                │  - Ranking preservado  │
                └───────────────────────┘
                                │
                                ▼
                ┌───────────────────────┐
                │  Retorna fitness       │
                │  Array (n_partículas,)  │
                │  Menor = Melhor        │
                └───────────────────────┘
```

---

## 🔗 Conexões Entre Componentes

### **1. PSO → PSOMCDMWrapper**

**Como funciona:**
```python
# Em pso.py
pso = PSO(population_size=20, ...)

# Cria wrapper com MCDM escolhido
wrapper = PSOMCDMWrapper(
    mcdm_method="TOPSIS",  # ou "OACE" ou "VIKOR"
    train_function=my_train_function,
    criteria_weights=[...],
    criteria_types=[...]
)

# PSO usa wrapper.fitness_function como função de fitness
pso.optimize(fitness_function=wrapper.fitness_function)
```

**O que acontece:**
- PSO chama `wrapper.fitness_function(batch_de_partículas)`
- Recebe array de fitness para atualizar pbest/gbest
- Não precisa saber qual MCDM está sendo usado

---

### **2. PSOMCDMWrapper → MCDMs**

**Como funciona:**
```python
# No wrapper, dentro de fitness_function():

# Para OACE (individual):
if evaluation_type == "individual":
    # Avalia cada partícula separadamente
    for each particle:
        score = oace.evaluate([particle_metrics])

# Para TOPSIS/VIKOR (contextual):
else:
    # Avalia matriz completa de uma vez
    scores = topsis.evaluate(decision_matrix)  # Matriz (n, 9)
```

**O que acontece:**
- Wrapper decide qual modo usar baseado em `EVALUATION_TYPES`
- Cria instância do MCDM apropriado no `__init__`
- Chama método `evaluate()` do MCDM com formato correto

---

### **3. PSOMCDMWrapper → Funções Auxiliares**

**Como funciona:**
```python
# No wrapper.fitness_function():

# Fase 2: Constrói matriz
decision_matrix = build_decision_matrix(all_metrics)
# Recebe: List[Dict] → Retorna: np.ndarray (n, 9)

# Fase 4: Converte scores
fitness = mcdm_to_fitness(scores, invert=True)
# Recebe: np.ndarray [0.9, 0.7, ...] → Retorna: [0.1, 0.3, ...]
```

**O que acontece:**
- Wrapper usa funções auxiliares para transformações de dados
- Separa responsabilidades: wrapper coordena, utils transforma
- Facilita manutenção e reutilização

---

### **4. train_function → Métricas → Matriz**

**Como funciona:**
```python
# Função de treinamento (fornecida pelo usuário)
def my_train_function(particle_position):
    # Treina modelo com arquitetura definida por position
    model = create_model(particle_position)
    metrics = train_and_evaluate(model)
    
    return {
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

# Wrapper coleta métricas de todas as partículas
all_metrics = [train_function(pos) for pos in particles]

# build_decision_matrix converte para formato estruturado
decision_matrix = build_decision_matrix(all_metrics)
```

**O que acontece:**
- `train_function` é chamada para cada partícula
- Retorna dicionário com métricas (formato flexível)
- `build_decision_matrix` padroniza para matriz NumPy

---

## 📊 Fluxo de Dados Completo

### **Exemplo Prático:**

**Entrada (PSO):**
```python
particles = np.array([
    [0.1, 0.2, 0.3, ...],  # Partícula 1 (posição no espaço de busca)
    [0.4, 0.5, 0.6, ...],  # Partícula 2
    [0.7, 0.8, 0.9, ...],  # Partícula 3
])
```

**Passo 1: Treinamento**
```python
# Para cada partícula
metrics_1 = train_function(particles[0])
# → {"top1_acc": 0.85, "total_params": 3e6, ...}

metrics_2 = train_function(particles[1])
# → {"top1_acc": 0.80, "total_params": 5e6, ...}

metrics_3 = train_function(particles[2])
# → {"top1_acc": 0.90, "total_params": 2e6, ...}

all_metrics = [metrics_1, metrics_2, metrics_3]
```

**Passo 2: Construção de Matriz**
```python
decision_matrix = build_decision_matrix(all_metrics)
# → np.array([
#     [0.85, 0.92, ..., 3e6, ...],  # Partícula 1
#     [0.80, 0.90, ..., 5e6, ...],  # Partícula 2
#     [0.90, 0.95, ..., 2e6, ...],  # Partícula 3
# ])
# Shape: (3, 9)
```

**Passo 3: Avaliação MCDM**

**Se TOPSIS:**
```python
topsis = TOPSIS(...)
result = topsis.evaluate(decision_matrix)
# → {"scores": [0.85, 0.72, 0.91]}
#   0.91 é melhor (partícula 3)
```

**Se OACE:**
```python
oace = OACE(...)
# Avalia uma por vez (mas usa mesma matriz para consistência)
scores = [0.82, 0.78, 0.88]
# 0.88 é melhor (partícula 3)
```

**Passo 4: Conversão para Fitness**
```python
mcdm_scores = [0.85, 0.72, 0.91]  # Maior = melhor
fitness = mcdm_to_fitness(mcdm_scores, invert=True)
# → [0.15, 0.28, 0.09]  # Menor = melhor
# 0.09 é melhor (partícula 3)
```

**Saída (PSO):**
```python
fitness = [0.15, 0.28, 0.09]
# PSO usa isso para atualizar:
# - pbest: melhor histórico de cada partícula
# - gbest: melhor global (partícula 3 com fitness 0.09)
```

---

## 🎯 Benefícios da Arquitetura

### **1. Separação de Responsabilidades**

- **`mcdm/utils.py`**: Funções puras de transformação (sem estado)
- **`PSOMCDMWrapper`**: Coordenação e orquestração
- **MCDMs**: Lógica de avaliação específica
- **PSO**: Algoritmo de otimização

### **2. Modularidade**

- Funções auxiliares podem ser reutilizadas em outros contextos
- Wrapper pode adicionar novos MCDMs facilmente
- PSO não precisa conhecer detalhes dos MCDMs

### **3. Comparação Justa**

- Todos os MCDMs recebem **mesmas partículas**
- Todos usam **mesma função de treinamento**
- Diferenças são apenas **metodológicas** (não técnicas)

### **4. Manutenibilidade**

- Cada função tem responsabilidade única e clara
- Fácil de testar individualmente
- Fácil de debugar (cada etapa é isolada)

---

## 🔍 Pontos Importantes

### **Por que `build_decision_matrix` é necessário?**

1. **Formato inconsistente**: Métricas vêm como dicionários (flexível)
2. **Formato necessário**: MCDMs precisam de matriz NumPy (estruturada)
3. **Padronização**: Garante ordem consistente dos critérios

### **Por que `mcdm_to_fitness` é necessário?**

1. **Incompatibilidade de sinais**: MCDMs retornam scores onde maior = melhor
2. **PSO minimiza**: Precisa de valores onde menor = melhor
3. **Ranking preservado**: A inversão mantém a ordem relativa

### **Por que está em `mcdm/utils.py` e não em `optimizers/`?**

- Funções são **genéricas** e podem ser usadas em outros contextos
- Pertencem ao domínio **MCDM** (transformação de dados de decisão)
- Mantém organização lógica (MCDM utils para MCDM)

---

## 📝 Resumo Executivo

### **O que foi criado:**

1. **`build_decision_matrix()`**: 
   - Converte List[Dict] → np.ndarray estruturado
   - Padroniza ordem dos critérios
   - Necessário para TOPSIS/VIKOR

2. **`mcdm_to_fitness()`**: 
   - Converte scores (maior=melhor) → fitness (menor=melhor)
   - Preserva ranking
   - Necessário para compatibilidade PSO

### **Como se conecta:**

```
PSO → Wrapper → Funções Auxiliares → MCDMs → Scores → Fitness → PSO
```

### **Fluxo de dados:**

```
Posições → Treinamento → Métricas (Dict) → Matriz (Array) → 
MCDM Scores → Fitness → PSO Updates
```

### **Benefícios:**

- ✅ Código modular e reutilizável
- ✅ Fácil adicionar novos MCDMs
- ✅ Comparação justa entre métodos
- ✅ Manutenção simplificada

---

**Tudo está conectado e funcionando como um sistema integrado!** 🎯

