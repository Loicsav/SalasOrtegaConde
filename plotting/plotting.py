"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, UCB1, Softmax


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, UCB1):
         label += f" (c={algo.c})"
    elif isinstance(algo, Softmax):
         label += f" (tau={algo.tau})"
    #Añadir más condiciones para otros algoritmos aquí
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm], optimal_reward: float = None):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param optimal_reward: Recompensa esperada del brazo óptimo (opcional).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    if optimal_reward is not None:
        plt.axhline(y=optimal_reward, color='black', linestyle='--', label='Recompensa Óptima', linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Porcentaje de Selección del Brazo Óptimo (%)', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats, algorithms):
    """
    Genera gráficos separados mostrando la selección de brazos y sus recompensas promedio.

    :param arm_stats: Lista de listas de diccionarios con estadísticas de cada brazo por algoritmo.
                      Cada diccionario contiene 'counts' (selecciones) y 'values' (recompensa promedio).
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    
    # Iterar sobre cada algoritmo
    for algo_idx, algo in enumerate(algorithms):
        # Crear figura para este algoritmo
        fig, ax = plt.subplots(figsize=(10, 6))
        arm_data = arm_stats[algo_idx]
        
        # Extraer datos de cada brazo
        num_arms = len(arm_data)
        arm_id = list(range(1, num_arms + 1))  # Brazos numerados del 1 al k
        mean_rewards = [arm['values'] for arm in arm_data]
        pulled_counts = [arm['counts'] for arm in arm_data]
        
        # Determinar el brazo óptimo (el que tiene mayor recompensa promedio estimada)
        optimal_arm_id = np.argmax(mean_rewards)
        optimal = [i == optimal_arm_id for i in range(num_arms)]
        
        # Calcular porcentaje de selecciones (basado en 1000 pasos por run, ajustable según datos)
        total_selections = sum(pulled_counts)
        selection_percentages = [(count / total_selections * 100) if total_selections > 0 else 0 
                                 for count in pulled_counts]
        
        # Etiquetas para el eje X con información detallada
        x_labels = [f"Brazo {id}\n{pulled_counts[i]} sel. ({round(selection_percentages[i], 1)}%) - {'Óptimo' if opt else 'No óptimo'}" 
                    for i, (id, opt) in enumerate(zip(arm_id, optimal))]
        
        # Definir colores: verde para óptimo, azul para no óptimo
        colors = ["#2ECC71" if opt else "#DB4D34" for opt in optimal]
        
        # Crear gráfico de barras
        ax.bar(x_labels, mean_rewards, color=colors)
        
        # Configuración de ejes y título
        ax.set_xlabel("Nº selecciones del Brazo")
        ax.set_ylabel("Promedio de Recompensas")
        algorithm_label = get_algorithm_label(algo)
        ax.set_title(f"Estadísticas de brazos - {algorithm_label}")
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
        
        # Mostrar valores en la consola
        print(f"\n{'='*60}")
        print(f"Estadísticas de brazos - {algorithm_label}")
        print(f"{'='*60}")
        for i, idx in enumerate(arm_id):
            optimal_mark = " (ÓPTIMO)" if optimal[i] else ""
            print(f"Brazo {idx}: Recompensa Promedio = {mean_rewards[i]:.4f}, "
                  f"Selecciones = {pulled_counts[i]}{optimal_mark}")
        
        # Ajustar diseño y mostrar
        plt.tight_layout()
        plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    # Si se proporcionan argumentos adicionales, se pueden graficar como referencia
    if args:
        for arg in args:
            plt.plot(range(steps), arg, label=f"Referencia: {arg}", linestyle='--')

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()