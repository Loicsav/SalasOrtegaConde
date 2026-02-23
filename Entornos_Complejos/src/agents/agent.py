from collections import defaultdict
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, env, seed: int):
        self.env = env
        np.random.seed(seed)  # Fija la semilla para NumPy


    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError("Este método debe ser implementado por la subclase.")

    @abstractmethod
    def update(self, state, action, reward, next_state):
        raise NotImplementedError("Este método debe ser implementado por la subclase.")