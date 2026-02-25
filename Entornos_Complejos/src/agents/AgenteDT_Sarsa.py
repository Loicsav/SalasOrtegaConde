from collections import defaultdict
import gymnasium as gym
import numpy as np
from src.agents.agent import Agent

class AgenteDT_Sarsa(Agent):
    def __init__(self, env, seed: int,  num_episodes: int = 1000, discount_factor: float = 1.0, epsilon: float = 0.1, decay: bool = False, decay_rate:float=1000.0):
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
            self.epsilon = max(0.1, min(1.0, self.decay_rate / (n + 1)))

        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    def update(self, state, action, reward, next_state, next_action, episode_state:int):
        """
        Actualiza la tabla Q utilizando la tupla (state, action, reward, next_state, next_action).
        
        Args:
            state: Estado actual del entorno
            action: Acción tomada en el estado actual
            reward: Recompensa recibida después de tomar la acción
            next_state: Estado siguiente al tomar la acción
            next_action: Acción que se tomará en el siguiente estado (para SARSA)
        """
        # Incrementamos el conteo de visitas para este par (state, action)
        self.visit_counts[(state, action)] += 1
        # Calculamos el paso de aprendizaje (learning rate) como 1 / N(state, action)
        self.alpha = min(1, 100.0 / self.visit_counts[(state, action)])


        # Si el episodio ha teminado (episode_state == 1) o truncado (episode_state == 2), El Q del siguinte estado es 0
        if episode_state != 0:
            # Si no es el primer estado del episodio, actualizamos el valor de Q para el estado anterior
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            return
        
        # Calculamos el valor objetivo para SARSA
        target = reward + self.discount_factor * self.Q[next_state][next_action]
        
        # Actualizamos la tabla Q utilizando la fórmula de actualización de SARSA
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])