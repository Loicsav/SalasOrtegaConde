import base64

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from IPython.display import HTML
from base64 import b64encode


import torch  # PyTorch: manejo de tensores y redes neuronales.
import torch.nn as nn  # Módulo para definir modelos de redes neuronales.
import torch.nn.functional as F  # Funciones de activación y utilidades de PyTorch.
import imageio  # Para crear el GIF a partir de los fotogramas.


def generar_video(env, Q, video_folder, num_episodes=1, seed=42):

    trigger = lambda episode_id: episode_id % 4 == 0 # small function that tells the recording env when to record, in our case always
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=trigger #, #(episode_id + 0) % 5 == 0, # Graba todos los episodios
        #name_prefix="lander_episode" # Prefijo para los nombres de los archivos de vídeo
        #name_prefix=lambda episode_id: f"lander_episodio_{(episode_id + 1):03d}_"  # Función lambda en name_prefix
    )

    #env = RecordVideo(env, './video')

    for episode in range(num_episodes):
        state, info = env.reset(seed=seed)
        episode_over = False

        while not episode_over:
            # Renderiza el frame actual, esto lo guardará en el archivo de video.
            env.render()

            # Selecciona una acción aleatoria del espacio de acciones disponible.
            # Esto simula la política del agente, que en este caso es aleatoria.
            action = np.argmax(Q[state, :])

            # Ejecuta un paso en el entorno con la acción seleccionada.
            # Obtiene la siguiente observación, la recompensa,
            # y si el episodio ha terminado (terminated) o ha sido truncado (truncated).
            state, reward, terminated, truncated, info = env.step(action)

            # Verifica si el episodio ha terminado, ya sea por terminación o truncamiento.
            episode_over = terminated or truncated


    env.close() # Importante cerrar el entorno, ¡esto finaliza la grabación de vídeo!
    print(f"Grabación de episodios completada. Vídeos guardados en la carpeta '{video_folder}'")

def mostrar_video(ruta_video, ancho=600):
  # Leer el video y convertirlo a base64
  video_file = open(ruta_video, "rb").read()
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  
  # Retornar el HTML para mostrarlo
  return HTML(f"""
  <video width="{ancho}" controls>
    <source src="{video_url}" type="video/mp4">
  </video>
  """)




def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    #env.reset()  # Reseteamos el entorno para mostrar la imagen inicial
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()

def plot(list_stats):
  # Creamos una lista de índices para el eje x
  indices = list(range(len(list_stats)))

  # Creamos el gráfico
  plt.figure(figsize=(6, 3))
  plt.plot(indices, list_stats)

  # Añadimos título y etiquetas
  plt.title('Proporción de recompensas')
  plt.xlabel('Episodio')
  plt.ylabel('Proporción')

  # Mostramos el gráfico
  plt.grid(True)
  plt.show()

# Define la función para mostrar el tamaño de los episodios
def plot_lengths(episode_lengths):
    indices = list(range(len(episode_lengths)))
    
    # Calculamos la media móvil (curva de tendencia)
    window_size = 50 
    moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')

    # Creamos el gráfico
    plt.figure(figsize=(10, 5))
    
    # Graficamos la longitud de cada episodio
    plt.plot(indices, episode_lengths, label='Longitud Episodio', alpha=0.3, color='blue')
    
    # Graficamos la tendencia
    # Ajustamos el rango de x para la media móvil.
    # Dado que mode='valid', el resultado tiene longitud N - K + 1.
    # El primer punto corresponde al promedio de los primeros K puntos (0 a K-1).
    # Lo alineamos al final de la ventana, es decir, x empieza en K-1.
    plt.plot(range(window_size-1, len(episode_lengths)), moving_avg, color='red', linewidth=2, label=f'Tendencia (Media móvil {window_size})')

    plt.title('Longitud de los episodios durante el entrenamiento')
    plt.xlabel('Episodio')
    plt.ylabel('Longitud (Pasos)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_two_lengths(episode_lengths1, episode_lengths2, window_size=50):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    def plot_one(ax, episode_lengths, title):
        indices = list(range(len(episode_lengths)))
        if len(episode_lengths) >= window_size:
            moving_avg = np.convolve(
                episode_lengths,
                np.ones(window_size)/window_size,
                mode='valid'
            )
            ax.plot(indices, episode_lengths,
                    label='Longitud Episodio',
                    alpha=0.3, color='blue')
            ax.plot(range(window_size-1, len(episode_lengths)),
                    moving_avg, color='red', linewidth=2,
                    label=f'Tendencia (Media mÃ³vil {window_size})')
        else:
            ax.plot(indices, episode_lengths,
                    label='Longitud Episodio',
                    alpha=0.3, color='blue')

        ax.set_title(title)
        ax.set_xlabel('Episodio')
        ax.grid(True)
        ax.legend()

    plot_one(ax1, episode_lengths1, 'Entrenamiento 1')
    plot_one(ax2, episode_lengths2, 'Entrenamiento 2')

    fig.suptitle('Longitud de los episodios durante el entrenamiento')
    ax1.set_ylabel('Longitud (Pasos)')  # eje Y compartido
    plt.tight_layout()
    plt.show()

def get_latest_episode_video_file(directory):
    # Expresión regular que coincide con el formato de los ficheros de video
    pattern = re.compile(r"rl-video-episode-(\d+)\.mp4")
    latest_file = None
    highest_episode = -1

    # Busca en el directorio
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            episode_number = int(match.group(1))  # Extrae el número de episodio
            # Comprobamos, para conseguir el número de episodio más alto.
            if episode_number > highest_episode:
                highest_episode = episode_number
                latest_file = os.path.join(directory, filename)  # Almacena el path completo

    return latest_file

def _state_to_tensor(state, device):
    """
    Convierte el estado a un vector One-Hot Encoding aplanado.
    """
    # 1. Aplanamos la matriz de tuplas a un array de enteros: (10, 2) -> (20,)
    s = np.asarray(state, dtype=np.int64).flatten() 
    
    # 2. Convertimos a tensor de PyTorch
    s_tensor = torch.from_numpy(s).to(device)
    
    # 3. Aplicamos One-Hot Encoding. Asumimos que los valores van de 0 a 19.
    # Esto genera una matriz de shape (20, 20)
    s_one_hot = F.one_hot(s_tensor, num_classes=20)
    
    # 4. Aplanamos todo a un solo vector de (1, 400) y lo pasamos a float
    # Es necesario pasarlo a float porque la red neuronal espera decimales para los pesos
    s_one_hot_flat = s_one_hot.view(1, -1).float()
    
    return s_one_hot_flat

def greedy_action_q_network(q_network, state, device):
    """
    Selecciona la acción óptima (greedy) para un estado dado utilizando la red Q.

    Parámetros:
      - q_network (QNetwork): Red neuronal con los pesos cargados.
      - state: Estado actual del entorno (puede ser una lista o tensor).
      - device: Dispositivo donde se ejecuta la red neuronal (CPU o GPU).

    Retorna:
      - int: Acción que maximiza 
.
    """
    # Desactivamos el cálculo de gradientes (no es necesario en modo evaluación).
    # Explotación
    state_t = _state_to_tensor(state, device)
    with torch.no_grad():
        q_values = q_network(state_t)
    return int(q_values.argmax(dim=1).item())

def greedy_action_tiling(q, state):
    """
    Selecciona la acción óptima (greedy) para un estado dado utilizando la red Q.

    Parámetros:
      - q (list of arrays): Valores Q para cada tilings.
      - state: Estado actual del entorno (puede ser una lista o tensor).

    Retorna:
      - int: Acción que maximiza 
.
    """
    av_list = []
    for k, idx in enumerate(state):
        av = q[k][idx]
        av_list.append(av)

    av = np.mean(av_list, axis=0)
    return np.random.choice(np.flatnonzero(av==av.max()))

def run_episode_greedy(env, q, seed=42, tipo_algoritmo="Tiling", max_steps=500, device=None):
    """
    Ejecuta un episodio usando la política greedy y captura los fotogramas.

    Parámetros:
      - env: Entorno Gymnasium configurado con render_mode='rgb_array'.
      - q (list of arrays): Valores Q para cada tilings.
      - tipo_algoritmo (str): Tipo de algoritmo ("Tiling" o "QNetwork").
      - max_steps (int): Número máximo de pasos a ejecutar en el episodio.

    Retorna:
      - list: Lista de fotogramas (imágenes) capturados durante el episodio.
    """
    frames = []  # Lista para almacenar cada fotograma.

    # Reiniciar el entorno y obtener el estado inicial.
    state, _ = env.reset(seed=seed)
    done = False  # Indicador de finalización del episodio.

    # Ejecutar el episodio hasta max_steps o hasta que el entorno indique que ha terminado.
    for _ in range(max_steps):
        # Capturar el fotograma actual del entorno.
        frame = env.render()
        frames.append(frame)

        # Seleccionar la acción óptima utilizando la función greedy.
        if tipo_algoritmo == "Tiling":
            action = greedy_action_tiling(q, state)
        elif tipo_algoritmo == "QNetwork":
            action = greedy_action_q_network(q, state, device)
        else:
            raise ValueError("tipo_algoritmo debe ser 'Tiling' o 'QNetwork'.")

        # Ejecutar la acción en el entorno y obtener el siguiente estado y otros datos.
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state  # Actualizar el estado.

        # Si el episodio ha terminado o se ha truncado, capturar el fotograma final y salir.
        if done or truncated:
            frames.append(env.render())
            break

    return frames


def frames_to_gif(frames, filename="cartpole_sarsa.gif"):
    """
    Crea un archivo GIF a partir de una lista de fotogramas.

    Parámetros:
      - frames (list): Lista de fotogramas (imágenes) capturados del entorno.
      - filename (str): Nombre del archivo GIF resultante.

    Retorna:
      - str: Nombre del archivo GIF creado.
    """
    # Abrir un escritor de GIF con imageio.
    with imageio.get_writer(filename, mode='I') as writer:
        # Agregar cada fotograma al GIF.
        for frame in frames:
            writer.append_data(frame)
    return filename

def display_gif(gif_path):
    """
    Muestra un GIF en Google Colab.

    Parámetros:
      - gif_path (str): Ruta del archivo GIF.

    Retorna:
      - HTML: Objeto HTML que contiene el GIF incrustado.
    """
    # Abrir el archivo GIF en modo binario.
    with open(gif_path, 'rb') as f:
        video = f.read()
    # Convertir el contenido del GIF a una cadena Base64.
    b64 = base64.b64encode(video)
    # Retornar el objeto HTML que muestra el GIF.
    return HTML(f'<img src="data:image/gif;base64,{b64.decode()}" style="border: 2px solid black;">')

def save_training_results_to_csv(filename, results):
    """
    Guarda los resultados del entrenamiento (recompensas promedio y longitudes de episodios)
    de varios algoritmos en un archivo CSV.

    Args:
        filename (str): Nombre del archivo donde se guardarán los datos (ej: 'resultados.csv').
        results (dict): Un diccionario donde las claves son los nombres de los algoritmos 
                        y los valores son diccionarios con claves 'stats' (lista de recompensas promedio)
                        y 'lengths' (lista de longitudes de episodio).
                        Ejemplo:
                        {
                            'SARSA': {'stats': [0.1, 0.2], 'lengths': [100, 90]},
                            'Q-Learning': {'stats': [0.1, 0.3], 'lengths': [100, 80]}
                        }
    """
    data_list = []
    
    for algo_name, metrics in results.items():
        stats = metrics.get('stats', [])
        lengths = metrics.get('lengths', [])
        
        # Determinar la longitud mínima para evitar errores si las listas no coinciden
        min_len = min(len(stats), len(lengths))
        
        for i in range(min_len):
            data_list.append({
                'Algorithm': algo_name,
                'Episode': i + 1,
                'Average_Reward': stats[i],
                'Episode_Length': lengths[i]
            })
            
    df = pd.DataFrame(data_list)
    df.to_csv(filename, index=False)
    print(f"Datos de entrenamiento guardados exitosamente en {filename}")