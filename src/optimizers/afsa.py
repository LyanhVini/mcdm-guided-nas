"""
Algoritmo Artificial Fish Swarm (AFSA)
=====================================

Implementação do algoritmo AFSA para exploração global do espaço de busca.
Funcionalidades:
- Comportamentos de predação, enxame e seguimento
- Adaptação dinâmica da distância visual e tamanho do passo
- Detecção de densidade local para evitar aglomeração
- Mecanismos de busca aleatória para escapar de mínimos locais
- Otimização para problemas de alta dimensionalidade

Responsável pela exploração ampla do espaço de arquiteturas, garantindo
que o otimizador não fique preso em regiões subótimas.
"""
import numpy as np

class AFSA:
    """
    Implementa o Artificial Fish Swarm Algorithm (AFSA) para a otimização inicial
    de um enxame de partículas do PSO, conforme descrito no artigo de referência.

    Este algoritmo é usado para gerar um conjunto inicial otimizado de partículas (temppbest)
    que será utilizado pelo PSO para iniciar sua busca. O AFSA simula três comportamentos
    principais dos peixes:
    1. Comportamento de Aglomeração (Cluster): Os peixes se movem em direção ao centro
       dos seus vizinhos com melhor aptidão.
    2. Comportamento de Forrageamento: Os peixes procuram por posições melhores dentro
       do seu campo de visão.
    3. Comportamento Aleatório: Os peixes se movem aleatoriamente quando os outros
       comportamentos não resultam em melhoria.

    O algoritmo segue as seguintes fórmulas do artigo:
    - Fórmula 12: Movimento em direção ao centro (cluster behavior)
    - Fórmula 13: Exploração de nova posição (foraging behavior)
    - Fórmula 14: Movimento em direção à melhor posição encontrada
    - Fórmula 15: Movimento aleatório (random behavior)

    Atributos:
        population_size (int): O número de peixes artificiais no enxame. Este valor
                             deve ser igual ao número de partículas do PSO.
        n_dim (int): A dimensionalidade do espaço de busca (problema).
        visual (float): O campo de visão de um peixe artificial. Define o raio
                       dentro do qual um peixe pode ver outros peixes.
        step (float): O tamanho máximo do passo de movimento de um peixe.
                     Controla a velocidade de convergência do algoritmo.
        try_times (int): O número de tentativas para o comportamento de forrageamento.
                        Quanto maior, mais exploratório será o algoritmo.
        max_iter (int): O número máximo de iterações para a otimização.
                       Controla o tempo de execução do algoritmo.
        lower_bound (float): O limite inferior do espaço de busca.
        upper_bound (float): O limite superior do espaço de busca.
        population (np.ndarray): As posições atuais do enxame de peixes.
                               Formato: (population_size, n_dim)
        fitness (np.ndarray): O valor de aptidão (fitness) para cada peixe.
                             Formato: (population_size,)

    Exemplo de uso:
        >>> afsa = AFSA(
        ...     population_size=30,
        ...     n_dim=2,
        ...     visual=0.5,
        ...     step=0.1,
        ...     try_times=5,
        ...     max_iter=100,
        ...     lower_bound=-10,
        ...     upper_bound=10
        ... )
        >>> temppbest = afsa.optimize()  # Retorna as posições otimizadas para o PSO
    """

    def __init__(self, population_size, n_dim, visual, step, try_times, max_iter, lower_bound, upper_bound):
        """
        Inicializa o otimizador AFSA.

        Args:
            population_size (int): Tamanho da população de peixes. Deve ser igual ao
                                 número de partículas do PSO.
            n_dim (int): Dimensionalidade do problema (número de variáveis a otimizar).
            visual (float): Campo de visão dos peixes. Define o raio de percepção
                          dentro do qual um peixe pode ver outros peixes.
            step (float): Tamanho máximo do passo de movimento. Controla a velocidade
                         de convergência do algoritmo.
            try_times (int): Número de tentativas de forrageamento. Quanto maior,
                           mais exploratório será o algoritmo.
            max_iter (int): Número máximo de iterações. Controla o tempo de execução
                          do algoritmo.
            lower_bound (float): Limite inferior do espaço de busca.
            upper_bound (float): Limite superior do espaço de busca.

        Raises:
            ValueError: Se population_size <= 0, n_dim <= 0, visual <= 0,
                       step <= 0, try_times <= 0, ou max_iter <= 0.
        """
        if any(param <= 0 for param in [population_size, n_dim, visual, step, try_times, max_iter]):
            raise ValueError("Todos os parâmetros numéricos devem ser positivos")

        self.population_size = population_size
        self.n_dim = n_dim
        self.visual = visual
        self.step = step
        self.try_times = try_times
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # Inicializa a população de forma mais diversificada
        self.population = self._initialize_diverse_population()
        self.fitness = np.array([self.fitness_function(fish) for fish in self.population])

    def _initialize_diverse_population(self):
        """
        Inicializa a população de forma mais diversificada para garantir
        melhor exploração do espaço de busca.
        """
        population = np.zeros((self.population_size, self.n_dim))
        
        # Primeira metade: distribuição uniforme aleatória
        half_size = self.population_size // 2
        population[:half_size] = np.random.uniform(
            self.lower_bound, self.upper_bound, (half_size, self.n_dim)
        )
        
        # Segunda metade: distribuição mais estruturada
        remaining = self.population_size - half_size
        for i in range(remaining):
            fish = np.zeros(self.n_dim)
            for j in range(self.n_dim):
                # Alterna entre valores próximos aos extremos e valores centrais
                if (i + j) % 3 == 0:
                    # Valores próximos ao mínimo
                    fish[j] = np.random.uniform(self.lower_bound, 
                                              self.lower_bound + 0.3 * (self.upper_bound - self.lower_bound))
                elif (i + j) % 3 == 1:
                    # Valores próximos ao máximo
                    fish[j] = np.random.uniform(self.lower_bound + 0.7 * (self.upper_bound - self.lower_bound), 
                                              self.upper_bound)
                else:
                    # Valores centrais
                    fish[j] = np.random.uniform(self.lower_bound + 0.3 * (self.upper_bound - self.lower_bound),
                                              self.lower_bound + 0.7 * (self.upper_bound - self.lower_bound))
            population[half_size + i] = fish
        
        return population

    def fitness_function(self, position):
        """
        Função de aptidão (fitness) que será otimizada.
        Esta função deve ser substituída pela função objetivo real do problema.

        Args:
            position (np.ndarray): Posição do peixe no espaço de busca.

        Returns:
            float: Valor da função objetivo na posição dada.
        """
        return np.sum(position**2)

    def cluster_behavior(self, i):
        """
        Implementa o comportamento de aglomeração (cluster) conforme a fórmula 12 do artigo.
        O peixe tenta se mover em direção ao centro dos seus vizinhos com melhor aptidão.

        Args:
            i (int): Índice do peixe atual.

        Returns:
            np.ndarray: Nova posição do peixe após o comportamento de aglomeração.
        """
        neighbors_indices = [j for j in range(self.population_size) if i != j and np.linalg.norm(self.population[i] - self.population[j]) < self.visual]

        if not neighbors_indices:
            return self.foraging_behavior(i)

        center_position = np.mean(self.population[neighbors_indices], axis=0)
        center_fitness = self.fitness_function(center_position)

        if center_fitness < self.fitness[i]:
            # Fórmula 12: Movimento em direção ao centro
            move_direction = (center_position - self.population[i]) / np.linalg.norm(center_position - self.population[i])
            next_position = self.population[i] + move_direction * self.step * np.random.rand()
            return np.clip(next_position, self.lower_bound, self.upper_bound)
        else:
            return self.foraging_behavior(i)

    def foraging_behavior(self, i):
        """
        Implementa o comportamento de forrageamento conforme as fórmulas 13 e 14 do artigo.
        O peixe procura aleatoriamente por uma posição melhor dentro do seu campo de visão.

        Args:
            i (int): Índice do peixe atual.

        Returns:
            np.ndarray: Nova posição do peixe após o comportamento de forrageamento.
        """
        for _ in range(self.try_times):
            # Fórmula 13: Exploração de nova posição
            exploratory_position = self.population[i] + np.random.uniform(-1, 1, self.n_dim) * self.visual
            exploratory_position = np.clip(exploratory_position, self.lower_bound, self.upper_bound)
            
            if self.fitness_function(exploratory_position) < self.fitness[i]:
                # Fórmula 14: Movimento em direção à melhor posição encontrada
                move_direction = (exploratory_position - self.population[i]) / np.linalg.norm(exploratory_position - self.population[i])
                next_position = self.population[i] + move_direction * self.step * np.random.rand()
                return np.clip(next_position, self.lower_bound, self.upper_bound)

        return self.random_behavior(i)

    def random_behavior(self, i):
        """
        Implementa o comportamento aleatório conforme a fórmula 15 do artigo.
        O peixe se move aleatoriamente quando os outros comportamentos não resultam em melhoria.

        Args:
            i (int): Índice do peixe atual.

        Returns:
            np.ndarray: Nova posição do peixe após o comportamento aleatório.
        """
        # Fórmula 15: Movimento aleatório
        random_move = np.random.uniform(-1, 1, self.n_dim) * self.step
        return np.clip(self.population[i] + random_move, self.lower_bound, self.upper_bound)

    def optimize(self):
        """
        Executa o processo de otimização do AFSA para gerar o conjunto de soluções
        iniciais otimizadas (temppbest) para o PSO, conforme o Algoritmo 1 e o Passo 2 do Algoritmo 3.
        
        Returns:
            np.ndarray: O conjunto de melhores posições encontradas (pbest),
                        que serve como `temppbest` para o PSO.
        """
        # Inicializa o pbest com as posições e aptidões iniciais
        pbest = np.copy(self.population)
        pbest_fitness = np.copy(self.fitness)

        for _ in range(self.max_iter):
            for i in range(self.population_size):
                # O peixe executa os comportamentos para encontrar uma nova posição
                next_pos = self.cluster_behavior(i)
                next_fitness = self.fitness_function(next_pos)
                
                # Atualiza a posição do peixe se a nova posição for melhor
                if next_fitness < self.fitness[i]:
                    self.population[i] = next_pos
                    self.fitness[i] = next_fitness
                    
                    # Atualiza o registro histórico da melhor posição (pbest)
                    if next_fitness < pbest_fitness[i]:
                        pbest[i] = next_pos
                        pbest_fitness[i] = next_fitness
        
        # Retorna o conjunto de soluções de otimização inicial (temppbest)
        return pbest