"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the binomial distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from src.arms.arm import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de ensayos de la distribución binomial.
        :param p: Probabilidad de éxito en cada ensayo.
        """
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución.
        """

        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10, p_min: float = 0.0, p_max: float = 1.0):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param n: Número de ensayos para todos los brazos.
        :param p_min: Valor mínimo de la probabilidad.
        :param p_max: Valor máximo de la probabilidad.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0 <= p_min <= 1 and 0 <= p_max <= 1, "Las probabilidades deben estar entre 0 y 1."
        assert p_min < p_max, "El valor de p_min debe ser menor que p_max."

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)

        arms = [ArmBinomial(n, p) for p in p_values]

        return arms
