# Passos Revisados para Implementação MCDM-PSO

## 📊 Análise do Código Existente

### ✅ **O que JÁ EXISTE e está PRONTO:**

#### **Fase 1: Estrutura Base** ✅ COMPLETO
1. ✅ **`PSOMCDMWrapper`** - Classe criada e funcional
   - Localização: `src/optimizers/pso_mcdm_wrapper.py`
   - Métodos principais implementados
   - Mapeamento de tipos (individual/contextual) pronto

2. ✅ **Interface de tipos** - Implementada
   - `EVALUATION_TYPES` mapeando métodos
   - Métodos estáticos `is_contextual()` e `is_individual()`

3. ✅ **`build_decision_matrix()`** - Função auxiliar criada
   - Localização: `src/mcdm/utils.py`
   - Converte List[Dict] → np.ndarray
   - Ordem padrão definida (DEFAULT_CRITERIA_ORDER)

4. ✅ **`mcdm_to_fitness()`** - Função auxiliar criada
   - Localização: `src/mcdm/utils.py`
   - Converte scores MCDM → fitness PSO
   - Inversão de sinais implementada

#### **Fase 2: Componentes do Sistema** ✅ PRONTOS
5. ✅ **MCDMs Implementados**
   - `OACE`: `src/mcdm/oace.py` - Funcional
   - `TOPSIS`: `src/mcdm/topsis.py` - Funcional
   - `VIKOR`: `src/mcdm/vikor.py` - Funcional

6. ✅ **Função de Treinamento**
   - `_warm_up_candidate()`: `src/optimizers/afsa_pso.py` linha 401
   - Recebe posição → retorna métricas (Dict)
   - Usa `warm_up_mobilenet`, `warm_up_cnn`, etc.
   - Formato: `{"top1_acc": 0.85, "total_params": 3e6, ...}`

7. ✅ **PSO já aceita fitness_function**
   - `pso.py` linha 131: `optimize(fitness_function=None, ...)`
   - Já chama `fitness_function(particles)` com batch
   - Retorna array de fitness values

8. ✅ **Função de avaliação de modelos**
   - `evaluate_model()`: `src/utils/evaluate_utils.py`
   - Retorna métricas no formato correto

---

## 🔄 **Passos REVISADOS - O que ainda precisa ser feito**

### **Fase 1: Ajustes no Wrapper** (Pequenos ajustes)

#### ✅ Passo 1-4: **JÁ FEITO** - Estrutura base completa

#### 🔧 Passo 5: **Ajustar `fitness_function` do wrapper para usar `_warm_up_candidate`**
**Status**: Parcialmente feito - precisa ajuste

**O que fazer:**
- O wrapper já tem `train_function` como parâmetro ✅
- Precisa adaptar para usar a assinatura correta de `_warm_up_candidate`
- `_warm_up_candidate` recebe `candidate_vector: np.ndarray` (1D)
- Retorna `Dict[str, float]` com métricas

**Ação:**
```python
# JÁ ESTÁ ASSIM no wrapper:
def fitness_function(self, particles_positions: np.ndarray) -> np.ndarray:
    all_metrics = []
    for i in range(n_particles):
        metrics = self.train_function(particles_positions[i])  # ✅ Correto
        all_metrics.append(metrics)
```

**✅ ESTÁ CORRETO - Nenhum ajuste necessário!**

---

### **Fase 2: Integração com PSO** (Principal trabalho)

#### 🔧 Passo 11: **Criar função helper para configurar wrapper**
**Status**: NÃO FEITO

**O que fazer:**
- Criar função que retorna configuração padrão de pesos e tipos
- Extrair de `afsa_pso.py` ou criar padrão
- Usar mesma configuração para todos os MCDMs (comparação justa)

**Localização sugerida**: `src/optimizers/mcdm_config.py` ou dentro do wrapper

**Código necessário:**
```python
def get_default_mcdm_config():
    """Retorna configuração padrão para MCDMs."""
    criteria_weights = [
        0.40, 0.15, 0.25, 0.15, 0.05,  # top1, top5, prec, rec, f1
        0.25, 0.25, 0.25, 0.25          # params, time, mem, gflops
    ]
    criteria_types = ["benefit"] * 5 + ["cost"] * 4
    return criteria_weights, criteria_types
```

#### 🔧 Passo 12: **Integrar wrapper no PSO ou criar classe wrapper do PSO**
**Status**: NÃO FEITO

**O que fazer:**
- Opção A: Modificar `PSO.__init__` para aceitar `mcdm_method`
- Opção B: Criar classe `PSOWithMCDM` que herda ou compõe PSO
- Opção C: Manter PSO como está e criar função helper que configura tudo

**Recomendação**: **Opção C** (menos invasiva)
- Criar função `create_pso_with_mcdm()` que:
  - Recebe `mcdm_method`, `train_function`, configurações
  - Cria wrapper
  - Cria PSO
  - Conecta wrapper.fitness_function ao PSO

**Estrutura:**
```python
def create_pso_with_mcdm(
    mcdm_method: str,
    train_function: Callable,
    population_size: int,
    n_dim: int,
    max_iter: int,
    # ... outros parâmetros PSO
):
    """Cria PSO configurado com MCDM."""
    # 1. Obter configuração MCDM
    weights, types = get_default_mcdm_config()
    mcdm_config = get_mcdm_specific_config(mcdm_method)
    
    # 2. Criar wrapper
    wrapper = PSOMCDMWrapper(
        mcdm_method=mcdm_method,
        train_function=train_function,
        criteria_weights=weights,
        criteria_types=types,
        mcdm_config=mcdm_config
    )
    
    # 3. Criar PSO
    pso = PSO(population_size, n_dim, max_iter, ...)
    
    # 4. Conectar
    pso.fitness_function = wrapper.fitness_function
    
    return pso, wrapper
```

#### 🔧 Passo 13: **Adaptar `_warm_up_candidate` para ser reutilizável**
**Status**: PARCIALMENTE FEITO

**Situação atual:**
- `_warm_up_candidate` está dentro de `AFSAPSO` (linha 401)
- Precisa ser acessível pelo wrapper

**Opções:**
- Opção A: Extrair `_warm_up_candidate` para função standalone em `utils/`
- Opção B: Criar classe base com `_warm_up_candidate` compartilhada
- Opção C: Passar `_warm_up_candidate` como `train_function` para wrapper

**Recomendação**: **Opção C** (mais simples)
- O wrapper já aceita `train_function` como parâmetro ✅
- Basta passar `afsa_pso._warm_up_candidate` ou criar wrapper function

**Código:**
```python
# No código que usa PSO:
afsa_pso = AFSAPSO(...)
wrapper = PSOMCDMWrapper(
    mcdm_method="TOPSIS",
    train_function=afsa_pso._warm_up_candidate,  # ✅ Usa função existente
    ...
)
```

---

### **Fase 3: Configuração e Parâmetros** (Ajustes)

#### 🔧 Passo 14: **Criar função de configuração de MCDM**
**Status**: NÃO FEITO

**O que fazer:**
- Função que retorna `mcdm_config` baseado no método
- Valores padrão para lambda (OACE), v (VIKOR), normalization_method

**Código:**
```python
def get_mcdm_specific_config(mcdm_method: str) -> Dict[str, Any]:
    """Retorna configuração específica para cada MCDM."""
    if mcdm_method == "OACE":
        return {"lambda_param": 0.5, "normalization_method": "min_max"}
    elif mcdm_method == "TOPSIS":
        return {"normalization_method": "vector", "distance_metric": "euclidean"}
    elif mcdm_method == "VIKOR":
        return {"v": 0.5, "normalization_method": "min_max"}
    else:
        return {}
```

#### ✅ Passo 15: **Função de inicialização** 
**Status**: PARCIALMENTE FEITO

**O que fazer:**
- Usar `get_default_mcdm_config()` e `get_mcdm_specific_config()`
- Criar instâncias consistentes dos MCDMs

---

### **Fase 4: Validação e Testes** (Novos)

#### 🔧 Passo 16: **Teste de integração wrapper + PSO**
**Status**: NÃO FEITO

**O que fazer:**
- Teste que usa wrapper com PSO real
- Valida que fitness é calculado corretamente
- Valida que ranking está correto

#### 🔧 Passo 17: **Teste comparativo simples**
**Status**: NÃO FEITO

**O que fazer:**
- Executar PSO com cada MCDM (mesma semente)
- Validar que métodos funcionam
- Não precisa comparar resultados ainda (isso é experimentação)

#### 🔧 Passo 18: **Logging para comparação**
**Status**: NÃO FEITO

**O que fazer:**
- Adicionar logs no wrapper indicando método usado
- Registrar scores MCDM antes da conversão para fitness
- Permitir análise posterior

---

### **Fase 5: Experimentação** (Futuro)

#### 📋 Passo 19-20: **Scripts de experimento**
**Status**: FUTURO (não necessário agora)

**O que fazer:**
- Criar script que executa PSO 3x (um para cada MCDM)
- Salva resultados
- Compara convergência e qualidade final

---

## 🎯 **Passos PRIORITÁRIOS para Implementar Agora**

### **Passo Prioritário 1: Função de Configuração**
**Criar**: `src/optimizers/mcdm_config.py`
- `get_default_mcdm_config()`: pesos e tipos padrão
- `get_mcdm_specific_config()`: configurações específicas por método

### **Passo Prioritário 2: Função Helper de Integração**
**Criar**: Função `create_pso_with_mcdm()` em `pso_mcdm_wrapper.py` ou novo arquivo
- Recebe parâmetros e retorna PSO configurado com wrapper
- Simplifica uso

### **Passo Prioritário 3: Adaptar uso existente**
**Modificar**: `afsa_pso.py` ou criar novo script
- Opção A: Adicionar parâmetro `mcdm_method` ao `AFSAPSO`
- Opção B: Criar novo script que usa PSO direto com wrapper

**Recomendação**: **Opção B** (não quebrar código existente)

### **Passo Prioritário 4: Teste de Integração**
**Criar**: Teste que valida funcionamento completo
- PSO + Wrapper + MCDM escolhido
- Valida formato de entrada/saída

---

## 📝 **Resumo: O que fazer agora**

### **✅ JÁ ESTÁ FEITO:**
- [x] Passos 1-4: Estrutura base (PSOMCDMWrapper, funções auxiliares)
- [x] Passos 5-10: Wrapper implementado (fitness_function, avaliação)
- [x] Passos 14-15: Parcialmente (MCDMs já têm configuração)

### **🔧 PRECISA FAZER:**
1. **Criar funções de configuração** (pesos, tipos, config específico)
2. **Criar função helper** `create_pso_with_mcdm()` 
3. **Adaptar/criar script** que usa PSO com wrapper
4. **Teste de integração** completo

### **📋 FUTURO:**
- Testes comparativos
- Scripts de experimentação
- Análise estatística

---

## 🔄 **Passos Simplificados e Revisados**

### **Etapa 1: Configuração** (1-2 horas)
1. Criar `src/optimizers/mcdm_config.py` com funções de configuração
2. Testar configurações isoladamente

### **Etapa 2: Integração** (2-3 horas)
3. Criar função `create_pso_with_mcdm()` 
4. Adaptar ou criar script que usa PSO com wrapper
5. Conectar `_warm_up_candidate` ao wrapper

### **Etapa 3: Testes** (1-2 horas)
6. Teste de integração básico
7. Validar que os 3 MCDMs funcionam com PSO

### **Etapa 4: Validação Final** (1 hora)
8. Executar exemplo completo
9. Verificar logs e resultados

**Total estimado**: 5-8 horas de desenvolvimento

---

## 🎯 **Próximos Passos Recomendados**

1. **AGORA**: Criar `mcdm_config.py` com funções de configuração
2. **DEPOIS**: Criar função helper de integração
3. **DEPOIS**: Adaptar script existente ou criar novo
4. **DEPOIS**: Testes de integração

**A estrutura base está 80% pronta!** Só falta conectar as peças finais. 🚀

