from collections import defaultdict
import gymnasium as gym
import numpy as np
from src.agents.agent import Agent

class AgentDT_QL(Agent):
    def __init__(self, env):
        self.env = env


    def get_action(self, state):
        raise NotImplementedError("Este método debe ser implementado por la subclase.")

    def update(self, state, action, reward, next_state):
        raise NotImplementedError("Este método debe ser implementado por la subclase.")