from queue import Queue
from collections import defaultdict
import gymnasium as gym
import numpy as np
from src.agents.agent import Agent
# Importamos las librerías necesarias
import torch  # PyTorch: manejo de tensores y redes neuronales.
import torch.nn as nn  # Módulo para definir modelos de redes neuronales.
import torch.nn.functional as F  # Funciones de activación y utilidades de PyTorch.


class QNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q.

    Parámetros:
      - state_dim (int): Dimensión del estado (para CartPole: 4).
      - action_dim (int): Número de acciones posibles (para CartPole: 2).
      - hidden_dim (int): Número de neuronas en las capas ocultas (por defecto: 64).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        # Primera capa: de estado a capa oculta de tamaño hidden_dim.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Segunda capa oculta.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Capa de salida: de hidden_dim a número de acciones.
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Propagación hacia adelante.

        Parámetro:
          - x (Tensor): Estado de entrada con forma [batch_size, state_dim].

        Retorna:
          - Tensor: Valores Q para cada acción, con forma [batch_size, action_dim].
        """
        # Aplicar la primera capa seguida de ReLU.
        x = F.relu(self.fc1(x))
        # Aplicar la segunda capa seguida de ReLU.
        x = F.relu(self.fc2(x))
        # Capa de salida sin activación, para obtener los valores Q.
        x = self.fc3(x)
        return x
     

class AgenteDT_DeepQL(Agent):
    def __init__(self, env, seed: int, num_episodes: int = 1000, discount_factor: float = 1.0, epsilon: float = 0.1, decay: bool = False, decay_rate:float=1000.0):
        super().__init__(env, seed)
        self.discount_factor = discount_factor  
        self.epsilon = epsilon
        self.decay = decay  
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        self.nA = env.action_space.n
        # Diccionarios para almacenar retornos y conteos
        self.visit_counts = defaultdict(int)  # {(state, action): count}
        self.qNetwork = QNetwork(env.observation_space.n, self.nA )  # Red neuronal para aproximar Q
        self.targetNetwork = QNetwork(env.observation_space.n, self.nA )  # Red objetivo para estabilidad

        self.d = Queue()  # Memoria de experiencia para el replay buffer
    

    def save_experience(self, state, action, reward, next_state, done):
        """
        Guarda la experiencia en la memoria de replay.

        Parámetros:
          - state (array): Estado actual.
          - action (int): Acción tomada.
          - reward (float): Recompensa recibida.
          - next_state (array): Estado siguiente.
          - done (bool): Indica si el episodio ha terminado.
        """
        if self.d.full():
            self.d.get()  # Eliminar la experiencia más antigua si la memoria está llena
        self.d.put((state, action, reward, next_state, done))

    def get_action(self, state, n:int):
        """
        Selecciona una acción utilizando una política epsilon-greedy basada en la tabla Q.
        
        Args:
            state: Estado actual del entorno
        Returns:
            action: Acción seleccionada
        """
        with torch.no_grad():
            if np.random.random() < self.epsilon:
                return np.random.randint(self.nA) # Selecciona una acción al azar
            else:
                # Convertir el estado a tensor si no lo es y añadir dimensión de batch.
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).unsqueeze(0)
                # Calcular los valores Q para el estado.
                q_values = self.qNetwork(state)
                # Seleccionar la acción que maximiza Q(s,a).
                action = torch.argmax(q_values).item()
                return action

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
        

        if episode_state != 0:
            # Si no es el primer estado del episodio, actualizamos el valor de Q para el estado anterior
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            return
        
        # Actualizamos Q utilizando la acción seleccionada por Q
        best_action = np.argmax(self.Q[next_state])  # Obtenemos la acción para el siguiente estado utilizando la política epsilon-greedy
        target = reward + self.discount_factor * self.Q[next_state][best_action]
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
