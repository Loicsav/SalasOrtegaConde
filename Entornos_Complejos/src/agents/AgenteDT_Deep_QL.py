from queue import Queue
from collections import defaultdict
from random import random
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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # Optimizador para entrenar la red Q
        self.loss = nn.MSELoss()  # Función de pérdida para entrenar la red Q

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
     

class AgenteDT_DobleDeepQL(Agent):
    def __init__(self, env, seed: int, num_episodes: int = 1000, discount_factor: float = 1.0, epsilon: float = 0.1, decay: bool = False, decay_rate:float=1000.0):
        super().__init__(env, seed)
        self.discount_factor = discount_factor  
        self.epsilon = epsilon
        self.decay = decay  
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        self.nA = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Configuración para usar GPU si está disponible
        # Diccionarios para almacenar retornos y conteos
        self.visit_counts = defaultdict(int)  # {(state, action): count}

        self.qNetwork = QNetwork(env.observation_space.shape[0], self.nA )  # Red neuronal para aproximar Q
        self.targetNetwork = QNetwork(env.observation_space.shape[0], self.nA )  # Red objetivo para estabilidad


        self.d = Queue(maxsize=10000)  # Memoria de experiencia para el replay buffer
    

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

    def update_target_network(self):
        """
        Actualiza la red objetivo con los pesos de la red principal.
        """
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    def _state_to_tensor(self, state):
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)  # (10,2) -> (1,20)
        return torch.from_numpy(s).to(self.device)

    def get_action(self, state, n:int):
        """
        Selecciona una acción usando política epsilon-greedy.
        """

        if np.random.random() < self.epsilon:
            # Exploración
            return np.random.randint(self.nA)
        else:
            # Explotación
            state_t = self._state_to_tensor(state)
            with torch.no_grad():
                q_values = self.qNetwork(state_t)
            return int(q_values.argmax(dim=1).item())


    def _sample_experience(self, batch_size=64):
        batch = random.sample(self.d.queue, min(batch_size, len(self.d.queue)))
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def update(self):
        """
        Actualiza la tabla Q utilizando la tupla (state, action, reward, next_state, next_action).
        
        Args:
            state: Estado actual del entorno
            action: Acción tomada en el estado actual
            reward: Recompensa recibida después de tomar la acción
            next_state: Estado siguiente al tomar la acción
            next_action: Acción que se tomará en el siguiente estado (para SARSA)
        """
        if len(self.d.queue) < 64:
            return  # no hay suficientes muestras
        
        states, actions, rewards, next_states, dones = self._sample_experience(64)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device).view(len(states), -1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device).view(len(next_states), -1)
        
        # Q(s, a) actual (predicción de la red online)
        q_vals = self.qNetwork(states).gather(1, actions)

        # Acciones óptimas según la red online en el siguiente estado
        with torch.no_grad():
            next_q_online = self.qNetwork(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Valores Q de esas acciones pero usando la red target
            next_q_target = self.targetNetwork(next_states)
            next_q_vals = next_q_target.gather(1, next_actions)

            # Cálculo del target Double DQN
            target = rewards + self.discount_factor * next_q_vals * (1 - dones)

        # Pérdida y paso de optimización
        loss = self.qNetwork.loss(q_vals, target)

        self.qNetwork.optimizer.zero_grad()
        loss.backward()
        self.qNetwork.optimizer.step()

