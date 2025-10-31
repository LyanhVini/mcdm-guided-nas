"""
Cálculo de Pesos AHP (Analytic Hierarchy Process)
===============================================

Implementação do método AHP para cálculo de pesos de critérios.
Funcionalidades:
- Construção de matrizes de comparação pareada
- Cálculo de autovalores e autovetores
- Verificação de consistência das comparações
- Cálculo da razão de consistência (CR)
- Normalização e validação de pesos

Utilizado para determinar pesos dos critérios de forma sistemática
e consistente nos métodos MCDM, baseado em comparações pareadas.
"""
import numpy as np
from pyDecision.algorithm import ahp_method

metrics_a = ["precision", "accuracy", "recall"]
metrics_c = ["mtp", "tpi", "ms"]
epsilon = 1e-5 # Parâmetro da padronização para evitar divisão por zero

### AHP Method ###

weight_derivation = 'geometric' # 'mean'; 'geometric' or 'max_eigen'

dataset_a = np.array([# Dataset for assertiveness metrics (P -> A -> R)
  #P      A      R
  [1  ,   5,     7   ],   #P
  [1/5,   1,     3   ],   #A
  [1/7,   1/3,   1   ],   #R
])
dataset_c = np.array([ # Dataset for cost metrics (MTP -> TPI -> MS)
  #MTP    TPI    MS
  [  1,     5,   7   ],   #MTP
  [1/5,     1,   3   ],   #TPI
  [1/7,   1/3,   1   ],   #MS
])

# Call AHP Function
def calculate_ahp_weights(dataset, weight_derivation):
    
    weights, rc = ahp_method(dataset, wd = weight_derivation)
    w1, w2, w3 = weights[0], weights[1], weights[2]

    w = [w1, w2, w3] 

    return w

wa = calculate_ahp_weights(dataset_a, weight_derivation)
wc = calculate_ahp_weights(dataset_c, weight_derivation)

print(wa)
print(wc)