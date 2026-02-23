from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
from src.agents.agent import Agent


class AgenteMC_OnPolicy(Agent):
    """
    Agente de Monte Carlo con política On-Policy (primeras visitas).
    Implementa el algoritmo de Monte Carlo para todas las visitas.
    """
    
    def __init__(self, env, seed: int, num_episodes: int = 1000, discount_factor: float = 1.0, epsilon: float = 0.1, decay: bool = False, decay_rate: float = 1000.0):
        """
        Inicializa el agente de Monte Carlo On-Policy.
        
        Args:
            env: Entorno de OpenAI Gym
            discount_factor: Factor de descuento para el cálculo de retornos
            epsilon: Parámetro de exploración epsilon-greedy
        """
        super().__init__(env, seed)
        self.discount_factor = discount_factor  
        self.epsilon = epsilon
        self.decay = decay  
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        # Diccionarios para almacenar retornos y conteos
        self.visit_counts = defaultdict(int)  # {(state, action): count}
        
        # Tabla Q 
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA])
    
    def get_action(self, state, n:int):
        """
        Selecciona una acción utilizando una política epsilon-greedy basada en la tabla Q.
        
        Args:
            state: Estado actual del entorno
        Returns:
            action: Acción seleccionada
        """
        if self.decay:
            self.epsilon = min(1.0, self.decay_rate / (n + 1))


        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.nA), p=pi_A)
    
    def update(self, episode: List[Tuple[int, int, float]]):
        """
        Actualiza la tabla Q utilizando el episodio completo.
        
        Args:
            episode: Lista de tuplas (state, action, reward) que representan el episodio completo
        """
        G = 0.0
        episode_return = 0.0 # Variable extra solo para mantener tus estadísticas originales
        
        # Iteramos el episodio al revés para calcular G_t eficientemente
        for (state, action, reward) in reversed(episode):
            # Calculamos el retorno desde el final hacia el principio
            G = reward + self.discount_factor * G
            
            # Actualizamos Q (All-Visit: actualizamos cada vez que vemos el estado-acción)
            self.visit_counts[(state, action)] += 1.0
            alpha = 1.0 / self.visit_counts[(state, action)]
            self.Q[state, action] += alpha * (G - self.Q[state, action])



