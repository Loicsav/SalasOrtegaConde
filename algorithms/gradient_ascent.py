"""
Module: algorithms/gradient_ascent.py
Description: Implementación del algoritmo Gradient Ascent (Gradient Bandit) para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class GradientAscent(Algorithm):

    def __init__(self, k: int, alpha: float = 0.1, use_baseline: bool = True):
        """
        Inicializa el algoritmo Gradient Ascent.

        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje para actualizar las preferencias.
        :param use_baseline: Si True, utiliza la recompensa media como línea de base para reducir varianza.
        """
        super().__init__(k)
        self.alpha = alpha
        self.use_baseline = use_baseline
        # Preferencias de cada brazo (inicialmente 0)
        self.preferences = np.zeros(k, dtype=float)
        # Línea de base (recompensa promedio acumulada)
        self.baseline = 0.0
        # Contador para actualizar la línea de base
        self.reward_sum = 0.0

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política softmax de las preferencias.
        
        La probabilidad de seleccionar cada brazo es proporcional a exp(H(a)):
        P(a) = exp(H(a)) / sum(exp(H(i)))
        
        donde H(a) es la preferencia del brazo.

        :return: índice del brazo seleccionado.
        """
        # Calcula las probabilidades usando la distribución softmax
        exp_preferences = np.exp(self.preferences - np.max(self.preferences))  # Sustrae el máximo para evitar overflow
        probabilities = exp_preferences / np.sum(exp_preferences)
        
        # Selecciona un brazo según las probabilidades
        chosen_arm = np.random.choice(self.k, p=probabilities)
        
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias basado en la recompensa obtenida.

        :param chosen_arm: Índice del brazo que fue tirado.
        :param reward: Recompensa obtenida.
        """
        # Actualiza el contador y la suma de recompensas
        self.counts[chosen_arm] += 1
        self.reward_sum += reward
        
        # Actualiza la línea de base (recompensa promedio)
        total_steps = np.sum(self.counts)
        self.baseline = self.reward_sum / total_steps
        
        # Calcula el término de error
        if self.use_baseline:
            error = reward - self.baseline
        else:
            error = reward
        
        # Recalcula las probabilidades
        exp_preferences = np.exp(self.preferences - np.max(self.preferences))
        probabilities = exp_preferences / np.sum(exp_preferences)
        
        # Actualiza las preferencias usando gradiente ascenso
        # Para el brazo seleccionado: H(a) += alpha * error * (1 - P(a))
        # Para los otros brazos: H(a) -= alpha * error * P(a)
        self.preferences[chosen_arm] += self.alpha * error * (1 - probabilities[chosen_arm])
        
        for arm in range(self.k):
            if arm != chosen_arm:
                self.preferences[arm] -= self.alpha * error * probabilities[arm]
