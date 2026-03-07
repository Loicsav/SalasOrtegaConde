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
import pandas as pd

from src.algorithms.algorithm import Algorithm
from src.algorithms.epsilon_greedy import EpsilonGreedy
from src.algorithms.ucb1 import UCB1
from src.algorithms.gradient_ascent import Softmax


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

def boxplot_estimaciones_brazos(all_runs_rewards, all_runs_counts, algorithms, true_rewards=None, custom_labels=None):
    sns.set_theme(style="whitegrid", context="talk")
    
    all_data = []
    
    for algo_idx, algo in enumerate(algorithms):
        # Establece una etiqueta para cada algoritmo para la leyenda
        if custom_labels:
            algo_label = custom_labels[algo_idx]
        elif hasattr(algo, 'epsilon'):
            algo_label = f"$\epsilon={algo.epsilon}$"
        elif hasattr(algo, 'c'):
            algo_label = f"$c={algo.c}$"
        else:
            algo_label = f"Algoritmo {algo_idx + 1}"
        
        # Estimaciones final de recompensa y número de ejecuciones de cada brazo para cada ejecución para el algoritmo
        runs_rewards = all_runs_rewards[algo_idx]
        runs_counts = all_runs_counts[algo_idx]
        
        # Por cada ejecución y brazo, se guarda su información
        for run_idx in range(len(runs_rewards)):
            for arm_idx in range(len(runs_rewards[run_idx])):
                all_data.append({
                    "Brazo": f"B{arm_idx + 1}",
                    "Brazo_idx": arm_idx,
                    "Estimación": runs_rewards[run_idx][arm_idx],
                    "Count": runs_counts[run_idx][arm_idx],
                    "Algoritmo": algo_label
                })
    # Se crea un dataframe con la información para su ploteo
    df = pd.DataFrame(all_data)
    
    # Se configura el gráfico
    plt.figure(figsize=(18, 9))
    n_algos = len(df['Algoritmo'].unique()) # Número de algoritmos distintos
    
    ax = sns.boxplot(
        data=df, 
        x="Brazo", 
        y="Estimación", 
        hue="Algoritmo", 
        palette="viridis", 
        showfliers=False, 
        width=0.8, 
        linewidth=1.5
    )

    # Se dibuja en verde los valores de recompensa real por cada brazo
    if true_rewards is not None:
        for i, true_val in enumerate(true_rewards):
            ax.hlines(y=true_val, xmin=i-0.4, xmax=i+0.4, colors='#2ECC71', linestyles='--', linewidth=1.5, zorder=10)

    # Añadir los count medios
    width = 0.8
    offsets = np.linspace(-width/2, width/2, n_algos + 1)
    offsets = (offsets[:-1] + offsets[1:]) / 2 # Se calcula el centro de cada número
    
    # Agrupamos por brazo y algoritmo y calculamos la media del número de selecciones
    stats = df.groupby(['Brazo_idx', 'Algoritmo'])['Count'].mean().reset_index()
    
    # Nombres de los algoritmos
    algo_order = df['Algoritmo'].unique()
    
    for arm_idx in range(len(df['Brazo'].unique())):
        for j, algo_name in enumerate(algo_order):
            # Media del número de selecciones
            mean_count = stats[(stats['Brazo_idx'] == arm_idx) & 
                               (stats['Algoritmo'] == algo_name)]['Count'].values[0]
            
            # Posición del texto (número de selecciones medio)
            pos_x = arm_idx + offsets[j]
            
            # Escribir el texto
            y_pos = ax.get_ylim()[0] # Parte inferior de la gráfica
            ax.text(pos_x, y_pos, f'{(mean_count/10):.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black', alpha=0.7)

    # Imprimir la gráfica
    plt.title("Comparativa de estimaciones de recompensa", fontsize=20, pad=30)
    plt.legend(title="Configuración", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats, algorithms, true_rewards):
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

        for i, true_val in enumerate(true_rewards):
            ax.hlines(y=true_val, xmin=i-0.4, xmax=i+0.4, colors='black', linestyles='--', linewidth=3, label='Real' if i == 0 else "")
        
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