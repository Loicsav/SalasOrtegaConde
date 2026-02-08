"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax (Boltzmann) para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo Softmax (Boltzmann).

        :param k: Número de brazos.
        :param tau: Parámetro de temperatura que controla la exploración.
                   Valores altos (tau > 1) favorecen más exploración.
                   Valores bajos (tau < 1) favorecen más explotación.
        """
        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política softmax.
        
        La probabilidad de seleccionar cada brazo es proporcional a exp(Q(a) / τ):
        P(a) = exp(Q(a) / τ) / Σ exp(Q(i) / τ)
        
        donde:
        - Q(a) es la recompensa promedio estimada del brazo
        - τ es el parámetro de temperatura

        :return: índice del brazo seleccionado.
        """
        # Calcula las probabilidades usando la distribución softmax
        # Sustrae el máximo para evitar problemas de overflow numérico
        scaled_values = self.values / self.tau
        
        # Estabilidad numérica
        max_value = np.max(scaled_values)
        exp_values = np.exp(scaled_values - max_value)
        chosen_arm = np.random.choice(self.k, p=exp_values / np.sum(exp_values))
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las estimaciones de recompensa para el brazo seleccionado.

        :param chosen_arm: índice del brazo seleccionado.
        :param reward: recompensa obtenida al seleccionar el brazo.
        """
        # Actualiza la recompensa promedio usando la fórmula incremental
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] = self.values[chosen_arm] + (1.0 / self.k) * (reward - self.values[chosen_arm])
