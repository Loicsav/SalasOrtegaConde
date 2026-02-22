"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 (Upper Confidence Bound) para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 2.0):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (balance entre explotación y exploración).
        """
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        
        El algoritmo calcula el índice de confianza superior (UCB) para cada brazo:
        UCB = Q(a) + c * sqrt(ln(t) / N(a))
        
        donde:
        - Q(a) es la recompensa promedio estimada del brazo
        - N(a) es el número de veces que se ha seleccionado el brazo
        - t es el número total de pasos (suma de todos los counts)
        - c es el parámetro de exploración

        :return: índice del brazo seleccionado.
        """
        # Asegurar que todos los brazos se seleccionan al menos una vez para evitar división por cero
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # Calcula el número total de pasos
        total_steps = np.sum(self.counts)
        
        # Inicializa un array para almacenar los índices UCB
        ucb_values = np.zeros(self.k)
        ucb_values = self.values + self.c * np.sqrt((2 * np.log(total_steps)) / self.counts)
        
        # Selecciona el brazo con el mayor índice UCB
        chosen_arm = np.argmax(ucb_values)
        
        return chosen_arm

