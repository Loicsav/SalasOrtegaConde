from collections import deque
from collections import defaultdict
import random
import gymnasium as gym
import numpy as np
from src.agents.agent import Agent
import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class QNetwork(nn.Module):
    # ... (Tu código de QNetwork se mantiene exactamente igual) ...
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AgenteDT_DobleDeepQL(Agent):
    def __init__(self, env, seed: int, num_episodes: int = 1000, discount_factor: float = 1.0, epsilon: float = 0.1, decay: bool = False, decay_rate:float=1000.0):
        super().__init__(env, seed)
        self.discount_factor = discount_factor  
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.decay = decay  
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        self.nA = env.action_space.n
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.visit_counts = defaultdict(int)

        # NOTA: Asegúrate de que env.observation_space.shape[0] devuelva 20 (el tamaño de tu array aplanado de 10 tuplas)
        state_dim_one_hot = 400 
        
        self.qNetwork = QNetwork(state_dim_one_hot, self.nA).to(self.device)
        self.targetNetwork = QNetwork(state_dim_one_hot, self.nA).to(self.device)
        self.update_target_network()

        self.d = deque(maxlen=10000)
    
    def save_experience(self, state, action, reward, next_state, done):
        self.d.append((state, action, reward, next_state, done))

    def update_target_network(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    def _state_to_tensor(self, state):
        """
        Convierte el estado a un vector One-Hot Encoding aplanado.
        """
        # 1. Aplanamos la matriz de tuplas a un array de enteros: (10, 2) -> (20,)
        s = np.asarray(state, dtype=np.int64).flatten() 
        
        # 2. Convertimos a tensor de PyTorch
        s_tensor = torch.from_numpy(s).to(self.device)
        
        # 3. Aplicamos One-Hot Encoding. Asumimos que los valores van de 0 a 19.
        # Esto genera una matriz de shape (20, 20)
        s_one_hot = F.one_hot(s_tensor, num_classes=20)
        
        # 4. Aplanamos todo a un solo vector de (1, 400) y lo pasamos a float
        # Es necesario pasarlo a float porque la red neuronal espera decimales para los pesos
        s_one_hot_flat = s_one_hot.view(1, -1).float()
        
        return s_one_hot_flat

    def get_action(self, state, n:int):
        if self.decay:
            self.epsilon = min(self.epsilon_init, self.decay_rate / (n + 1))

        if np.random.random() < self.epsilon:
            return np.random.randint(self.nA)
        else:
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.qNetwork(state_tensor)
                action = torch.argmax(q_values).item()
            return action

    # --- NUEVOS MÉTODOS COMPLETADOS ---

    def _sample_experience(self, batch_size):
        # Con deque, sample es directo y ultrarrápido
        batch = random.sample(self.d, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 1. Convertimos toda la tupla de estados en un solo array de NumPy de golpe y aplanamos (64, 20)
        states_np = np.array(states, dtype=np.int64).reshape(batch_size, -1)
        next_states_np = np.array(next_states, dtype=np.int64).reshape(batch_size, -1)
        
        # 2. Enviamos al dispositivo (GPU/CPU) de una sola vez
        states_tensor = torch.from_numpy(states_np).to(self.device)
        next_states_tensor = torch.from_numpy(next_states_np).to(self.device)
        
        # 3. Aplicamos One-Hot a todo el bloque y aplanamos (batch_size, 400)
        states_tensor = F.one_hot(states_tensor, num_classes=20).view(batch_size, -1).float()
        next_states_tensor = F.one_hot(next_states_tensor, num_classes=20).view(batch_size, -1).float()
        
        # 4. Resto de tensores
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def update(self):
        """
        Actualiza la red Q utilizando un lote de experiencias (Double DQN).
        """
        # Usamos len() porque self.d es de tipo deque
        if len(self.d) < 64:
            return
            
        states, actions, rewards, next_states, dones = self._sample_experience(64)

        # 1. Obtener los valores Q actuales Q(s, a) desde la red principal
        current_q_values = self.qNetwork(states).gather(1, actions)

        # 2. Lógica de Double DQN para calcular el valor objetivo (target)
        with torch.no_grad():
            # A) La red PRINCIPAL decide cuál es la mejor acción en el estado siguiente
            best_next_actions = self.qNetwork(next_states).argmax(dim=1, keepdim=True)
            
            # B) La red OBJETIVO evalúa el valor Q de esa acción específica
            next_q_values = self.targetNetwork(next_states).gather(1, best_next_actions)
            
            # C) Calcular el objetivo con la ecuación de Bellman (si don=1, next_q_values se anula)
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        # 3. Calcular la pérdida usando MSE
        loss = self.qNetwork.loss(current_q_values, target_q_values)

        # 4. Optimizar los pesos de la red principal
        self.qNetwork.optimizer.zero_grad()
        loss.backward()
        
        # Opcional pero muy recomendado: Recortar gradientes para evitar inestabilidad
        torch.nn.utils.clip_grad_norm_(self.qNetwork.parameters(), max_norm=1.0)
        
        self.qNetwork.optimizer.step()

    # ----------------------------------

    def save(self, path):
        torch.save(self.qNetwork.state_dict(), path)

    def load(self, path):
        self.qNetwork.load_state_dict(torch.load(path, map_location=self.device))
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())