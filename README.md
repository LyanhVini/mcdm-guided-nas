# OACE-Optimizer: Neural Architecture Search com MCDM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Um framework inovador para Neural Architecture Search (NAS) que utiliza métodos de Tomada de Decisão Multicritério (MCDM) como função de fitness para guiar o otimizador híbrido AFSA-PSO.

## 🎯 Objetivo

Este projeto visa comparar diferentes métodos MCDM (OACE, TOPSIS e VIKOR) como funções de fitness para otimização de arquiteturas neurais, especificamente para classificação de imagens no dataset CIFAR-10.

## 🚀 Características Principais

- **Otimizador Híbrido AFSA-PSO**: Combina Artificial Fish Swarm Algorithm (AFSA) com Particle Swarm Optimization (PSO)
- **Métodos MCDM**: Implementação completa de OACE, TOPSIS e VIKOR
- **Arquiteturas Modulares**: Suporte a ResNet, MobileNet, EfficientNet e CNNs personalizadas
- **Avaliação Automática**: Treinamento e avaliação automática de arquiteturas neurais
- **Sistema de Logging**: Rastreamento detalhado do processo de otimização
- **Configuração Centralizada**: Sistema de configuração baseado em Python
- **Utilitários Especializados**: Ferramentas para dados, treinamento e avaliação

## 📁 Estrutura do Projeto

```
mcdm-guided-nas/
├── config.py                     # Configurações globais do projeto
├── main.py                       # Script principal de execução
├── requirements.txt              # Dependências do projeto
├── README.md                     # Documentação do projeto
├── results/                      # Diretório para resultados dos experimentos
│   └── .gitkeep
└── src/                          # Código fonte principal
    ├── mcdm/                     # Métodos de Tomada de Decisão Multicritério
    │   ├── __init__.py
    │   ├── oace.py              # Método OACE (Optimized Aggregation by Comprehensive Evaluation)
    │   ├── topsis.py            # Método TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    │   ├── vikor.py             # Método VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)
    │   └── utils.py             # Utilitários para métodos MCDM
    ├── models/                   # Arquiteturas de Redes Neurais
    │   ├── architecture_loader.py  # Carregador de arquiteturas
    │   ├── cnn/                 # Arquiteturas CNN personalizadas
    │   ├── efficientnet/        # Implementações EfficientNet
    │   ├── mobilenet/           # Implementações MobileNet
    │   └── resnet/              # Implementações ResNet
    ├── optimizer/               # Algoritmos de Otimização
    │   ├── __init__.py
    │   ├── afsa_pso.py         # Otimizador híbrido AFSA-PSO
    │   ├── afsa.py             # Algoritmo Artificial Fish Swarm
    │   └── pso.py              # Algoritmo Particle Swarm Optimization
    └── utils/                   # Utilitários Gerais
        ├── __init__.py
        ├── ahp_weights.py      # Cálculo de pesos AHP
        ├── data_loader.py      # Carregamento e preparação de dados
        ├── evaluation_utils.py # Utilitários de avaliação
        ├── optimization_logger.py # Sistema de logging
        └── training_utils.py   # Utilitários de treinamento
```

## 🛠️ Instalação

### Pré-requisitos

- Python 3.8 ou superior
- TensorFlow 2.12 ou superior
- CUDA (opcional, para aceleração GPU)

### Instalação das Dependências

```bash
# Clonar o repositório
git clone <repository-url>
cd mcdm-guided-nas

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

## 🚀 Uso Rápido

### 1. Executar Experimento Completo

```bash
# Usar configuração padrão
python main.py

# Usar configuração personalizada
python main.py --config config.py

# Executar apenas um método MCDM
python main.py --method OACE

# Configurações personalizadas
python main.py --iterations 50 --population 20 --method TOPSIS
```

### 2. Avaliação Rápida

```bash
# Executar com menos epochs para teste rápido
python main.py --quick --iterations 10
```

### 3. Análise de Resultados

```bash
# Os resultados são salvos automaticamente em results/
# Para análise, use as funções de visualização disponíveis
python -c "from src.utils.evaluation_utils import analyze_results; analyze_results('results/')"
```

## ⚙️ Configuração

### Arquivo de Configuração Principal

O arquivo `config.py` contém todas as configurações globais do projeto:

```python
# Configurações do Dataset
DATASET_CONFIG = {
    "name": "CIFAR-10",
    "num_classes": 10,
    "input_shape": [32, 32, 3],
    "batch_size": 128
}

# Configurações do Otimizador AFSA-PSO
OPTIMIZER_CONFIG = {
    "population_size": 30,
    "max_iterations": 100,
    "afsa": {
        "visual_distance": 2.5,
        "step_size": 0.5
    },
    "pso": {
        "inertia_weight": 0.9,
        "cognitive_weight": 2.0,
        "social_weight": 2.0
    }
}

# Configurações dos Métodos MCDM
MCDM_CONFIG = {
    "methods": ["OACE", "TOPSIS", "VIKOR"],
    "criteria": [
        {"name": "accuracy", "weight": 0.4, "type": "benefit"},
        {"name": "efficiency", "weight": 0.3, "type": "benefit"},
        {"name": "complexity", "weight": 0.2, "type": "cost"},
        {"name": "training_time", "weight": 0.1, "type": "cost"}
    ]
}
```

### Parâmetros da Linha de Comando

| Parâmetro | Descrição | Exemplo |
|-----------|-----------|---------|
| `--config` | Arquivo de configuração | `--config config.py` |
| `--method` | Método MCDM específico | `--method OACE` |
| `--iterations` | Número de iterações | `--iterations 50` |
| `--population` | Tamanho da população | `--population 20` |
| `--output` | Diretório de saída | `--output results/my_experiment` |
| `--quick` | Avaliação rápida | `--quick` |

## 🏗️ Arquitetura Modular

### Módulos Principais

#### 1. **MCDM** (`src/mcdm/`)
- **OACE**: Método de agregação otimizada com análise de consistência
- **TOPSIS**: Técnica de proximidade à solução ideal
- **VIKOR**: Método de compromisso para tomada de decisão
- **Utils**: Funções auxiliares para normalização e cálculo de pesos

#### 2. **Otimizador** (`src/optimizer/`)
- **AFSA-PSO**: Algoritmo híbrido principal
- **AFSA**: Implementação do Artificial Fish Swarm Algorithm
- **PSO**: Implementação do Particle Swarm Optimization

#### 3. **Modelos** (`src/models/`)
- **Architecture Loader**: Carregador unificado de arquiteturas
- **CNN**: Implementações de redes convolucionais personalizadas
- **ResNet**: Arquiteturas ResNet otimizadas
- **MobileNet**: Arquiteturas MobileNet para dispositivos móveis
- **EfficientNet**: Arquiteturas EfficientNet para eficiência

#### 4. **Utilitários** (`src/utils/`)
- **Data Loader**: Carregamento e preparação de datasets
- **Training Utils**: Ferramentas de treinamento e callbacks
- **Evaluation Utils**: Métricas e análise de performance
- **AHP Weights**: Cálculo de pesos usando Analytic Hierarchy Process
- **Optimization Logger**: Sistema de logging especializado

### Vantagens da Arquitetura Modular

- **Flexibilidade**: Fácil adição de novos métodos MCDM e arquiteturas
- **Reutilização**: Componentes podem ser usados independentemente
- **Manutenibilidade**: Código organizado e bem documentado
- **Extensibilidade**: Interface clara para extensões futuras
- **Testabilidade**: Cada módulo pode ser testado isoladamente

## 📊 Métodos MCDM Implementados

### 1. OACE (Optimized Aggregation by Comprehensive Evaluation)
- Combina agregação ponderada com análise de consistência
- Parâmetros: `alpha` (balanceamento), `beta` (consistência)
- Ideal para: Problemas com múltiplos critérios conflitantes

### 2. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
- Identifica alternativa mais próxima da solução ideal
- Parâmetros: Métrica de distância, método de normalização
- Ideal para: Problemas com soluções ideais claras

### 3. VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)
- Método de compromisso entre benefício máximo e arrependimento mínimo
- Parâmetros: `v` (estratégia de grupo)
- Ideal para: Problemas que requerem soluções de compromisso

## 📈 Análise de Resultados

### Análise de Resultados

O sistema fornece ferramentas integradas para análise:

1. **Comparação entre Métodos**: Gráficos comparativos de performance
2. **Curvas de Convergência**: Evolução da otimização ao longo das iterações
3. **Análise de Arquiteturas**: Características das melhores arquiteturas encontradas
4. **Visualizações Interativas**: Gráficos Plotly para análise detalhada
5. **Análise Estatística**: Testes estatísticos e correlações
6. **Relatórios Automáticos**: Geração automática de relatórios

### Estrutura dos Resultados

```
results/experiment_YYYYMMDD_HHMMSS/
├── method_comparison.json        # Comparação entre métodos
├── OACE/
│   ├── optimization_results.json # Resultados do OACE
│   └── config.py                # Configurações usadas
├── TOPSIS/
│   ├── optimization_results.json
│   └── config.py
└── VIKOR/
    ├── optimization_results.json
    └── config.py
```

## 🔬 Exemplos de Uso

### Exemplo 1: Experimento Básico

```python
from src.optimizer import AFSA_PSO_Optimizer
from src.utils.data_loader import DataLoader
from src.mcdm import OACE

# Configurar carregador de dados
data_loader = DataLoader(dataset_name="CIFAR-10")

# Configurar método MCDM
oace = OACE(
    criteria_weights=[0.4, 0.3, 0.2, 0.1],
    criteria_types=["benefit", "benefit", "cost", "cost"]
)

# Configurar otimizador
optimizer = AFSA_PSO_Optimizer(
    search_space_dim=20,
    population_size=30,
    max_iterations=100
)

# Executar otimização
results = optimizer.optimize()
```

### Exemplo 2: Análise de Sensibilidade

```python
# Análise de sensibilidade do OACE
sensitivity_results = oace.calculate_sensitivity_analysis(
    decision_matrix=decision_matrix,
    criteria_weights=weights,
    criteria_types=types,
    weight_changes=[-0.2, -0.1, 0.1, 0.2]
)
```

### Exemplo 3: Comparação de Métodos

```python
from src.mcdm import OACE, TOPSIS, VIKOR

methods = {
    "OACE": OACE(criteria_weights=weights),
    "TOPSIS": TOPSIS(criteria_weights=weights),
    "VIKOR": VIKOR(criteria_weights=weights)
}

results = {}
for name, method in methods.items():
    results[name] = method.evaluate(decision_matrix)
```

## 🧪 Testes e Validação

### Executar Testes

```bash
# Executar todos os testes
pytest tests/

# Executar testes específicos
pytest tests/test_optimizer.py
pytest tests/test_mcdm.py
```

### Validação de Resultados

```bash
# Validar implementação dos métodos MCDM
python -m pytest tests/test_mcdm_validation.py -v

# Testar otimizador com problema benchmark
python -m pytest tests/test_optimizer_benchmark.py -v
```

## 📚 Documentação da API

### Classe AFSA_PSO_Optimizer

```python
class AFSA_PSO_Optimizer:
    def __init__(self, search_space_dim, population_size, max_iterations, 
                 bounds=None, fitness_function=None, config=None):
        """
        Inicializa o otimizador híbrido AFSA-PSO.
        
        Args:
            search_space_dim: Dimensão do espaço de busca
            population_size: Tamanho da população
            max_iterations: Número máximo de iterações
            bounds: Limites do espaço de busca
            fitness_function: Função de fitness
            config: Configurações adicionais
        """
    
    def optimize(self, fitness_function=None, verbose=True):
        """
        Executa a otimização.
        
        Returns:
            Dicionário com resultados da otimização
        """
```

### Classe DataLoader

```python
class DataLoader:
    def __init__(self, dataset_name="CIFAR-10", config=None):
        """
        Inicializa o carregador de dados.
        
        Args:
            dataset_name: Nome do dataset
            config: Configurações de carregamento
        """
    
    def load_data(self, batch_size=128, validation_split=0.2):
        """
        Carrega e prepara os dados.
        
        Returns:
            Tupla com (train_loader, val_loader, test_loader)
        """
```

## 🤝 Contribuição

### Como Contribuir

1. **Fork** o repositório
2. **Crie** uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. **Commit** suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. **Push** para a branch (`git push origin feature/nova-feature`)
5. **Abra** um Pull Request

### Diretrizes de Contribuição

- Siga o padrão de código PEP 8
- Adicione testes para novas funcionalidades
- Atualize a documentação conforme necessário
- Use commits descritivos

### Áreas de Contribuição

- [ ] Implementação de novos métodos MCDM
- [ ] Otimizações de performance
- [ ] Suporte a novos datasets
- [ ] Melhorias na visualização
- [ ] Documentação e exemplos

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Contato

- **Autor**: [Seu Nome]
- **Email**: [seu.email@exemplo.com]
- **Projeto**: [Link para o repositório]

## 🙏 Agradecimentos

- TensorFlow/Keras pela infraestrutura de deep learning
- Comunidade científica pelos métodos MCDM implementados
- Contribuidores e testadores do projeto

## 📖 Referências

1. **AFSA**: Li, X., Shao, Z., & Qian, J. (2002). An optimizing method based on autonomous animats: fish-swarm algorithm.
2. **PSO**: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
3. **OACE**: [Referência do método OACE]
4. **TOPSIS**: Hwang, C. L., & Yoon, K. (1981). Multiple attribute decision making: methods and applications.
5. **VIKOR**: Opricovic, S. (1998). Multicriteria optimization of civil engineering systems.

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!**
