from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
from src.agents.agent import Agent


class AgenteMC_OffPolicy(Agent):
    """
    Agente de Monte Carlo con política Off-Policy (primeras visitas).
    Implementa el algoritmo de Monte Carlo para todas las visitas.
    """
    
    def __init__(self, env, seed: int,  num_episodes: int = 1000, discount_factor: float = 1.0, epsilon: float = 0.1, decay: bool = False, decay_rate: float = 1000.0 ):
        """
        Inicializa el agente de Monte Carlo Off-Policy.
        
        Args:
            env: Entorno de OpenAI Gym
            discount_factor: Factor de descuento para el cálculo de retornos
            epsilon: Parámetro de exploración epsilon-greedy para la política de comportamiento
        """
        super().__init__(env, seed)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decay = decay
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        
        # C(s,a): Suma de los pesos de importancia acumulados
        self.C = np.zeros([env.observation_space.n, env.action_space.n])
        
        # Tabla Q inicializada arbitrariamente (ceros está bien)
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA])
        
        # Política objetivo (greedy determinsita)
        # Se puede derivar implícitamente de Q, pero a veces es útil tenerla explícita.
        self.policy = np.zeros(env.observation_space.n, dtype=int)  # Inicialmente todas las acciones son 0, se actualizará a medida que aprendamos
        # Aquí usaremos np.argmax(self.Q[state]) directamente.

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
        best_action =  self.policy[state]
        pi_A[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    def update(self, episode: List[Tuple[int, int, float]]):
        """
        Actualiza la tabla Q utilizando Off-Policy MC Control.
        
        Args:
            episode: Lista de tuplas (state, action, reward) generadas por la política de comportamiento.
        """
        G = 0.0
        W = 1.0  # Peso de importancia
        
        # Iteramos el episodio al revés
        # episode[t] = (S_t, A_t, R_{t+1})
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            
            G = self.discount_factor * G + reward
            
            self.C[state, action] += W
            
            # Actualización incremental ponderada:
            # Q(S,A) := Q(S,A) + (W / C(S,A)) * (G - Q(S,A))
            self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
            
            # Política objetivo (Target Policy) es Greedy
            self.policy[state] = np.argmax(self.Q[state])
            
            # Si la acción tomada NO fue la mejor según la política objetivo,
            # entonces la probabilidad de esa acción en la target policy es 0.
            # Por tanto, W * (pi(a|s) / b(a|s)) será 0 (ya que pi(a|s)=0).
            if action != self.policy[state]:
                break

            #Actualizamos el factor de importancia W
            W = W * (1.0 / ((1.0 - self.epsilon) + (self.epsilon / self.nA)))  # b(a|s) para la acción tomada, que es la probabilidad de tomar esa acción bajo la política de comportamiento epsilon-greedy.
            



