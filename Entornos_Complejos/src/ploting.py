import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np


def generar_video(env, Q, video_folder, num_episodes=1, seed=42):

    trigger = lambda episode_id: episode_id % 4 == 0 # small function that tells the recording env when to record, in our case always
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=trigger #, #(episode_id + 0) % 5 == 0, # Graba todos los episodios
        #name_prefix="lander_episode" # Prefijo para los nombres de los archivos de vídeo
        #name_prefix=lambda episode_id: f"lander_episodio_{(episode_id + 1):03d}_",  # Función lambda en name_prefix
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



import seaborn as sns
import matplotlib.pyplot as plt

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