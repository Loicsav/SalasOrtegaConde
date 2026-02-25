from collections import defaultdict
import gymnasium as gym
import numpy as np
from src.agents.agent import Agent

class AgenteDT_SarsaSG(Agent):
    def __init__(self, env, seed: int,  num_episodes: int = 1000, discount_factor: float = 1.0, epsilon: float = 0.1, decay: bool = False, decay_rate:float=1000.0):
        super().__init__(env, seed)
        self.discount_factor = discount_factor  
        self.epsilon = epsilon
        self.decay = decay  
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        self.nA = env.action_space.n
        self.epsilon_init = epsilon
        # Diccionarios para almacenar retornos y conteos

        self.visit_counts = np.zeros((self.env.n, self.env.bins[0],self.env.bins[1], self.nA))
        
        # Tabla Q 

        self.action_values_Q = np.zeros((self.env.n, self.env.bins[0],self.env.bins[1], self.nA))

    def get_action(self, state, n:int):
        """
        Selecciona una acción utilizando una política epsilon-greedy basada en la tabla Q.
        
        Args:
            state: Estado actual del entorno
        Returns:
            action: Acción seleccionada
        """
        if self.decay:
            self.epsilon = min(self.epsilon_init, self.decay_rate / (n + 1))

        if np.random.random() < self.epsilon:
            return np.random.randint(self.nA) # Selecciona una acción al azar
        else:
            av_list = []
            for k, idx in enumerate(state):
                av = self.action_values_Q[k][idx]
                av_list.append(av)

            av = np.mean(av_list, axis=0)
            return np.random.choice(np.flatnonzero(av==av.max()))

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
        for k in range(self.env.n):
            self.visit_counts[k][state[k]][action] += 1

        for k in range(self.env.n):
            # Calculamos el paso de aprendizaje (learning rate) como 1 / N(k, state, action)
            self.alpha = min(1, 100.0 / self.visit_counts[k][state[k]][action])
            if episode_state != 0:
                # Si no es el primer estado del episodio, actualizamos el valor de Q para el estado anterior
                self.action_values_Q[k][state[k]][action] += self.alpha * (reward - self.action_values_Q[k][state[k]][action])
                continue

            qsa = self.action_values_Q[k][state[k]][action]
            next_qsa = self.action_values_Q[k][next_state[k]][next_action]
            self.action_values_Q[k][state[k]][action] += self.alpha * (reward + self.discount_factor * next_qsa - qsa)

