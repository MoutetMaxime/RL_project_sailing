from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from tqdm import tqdm

from agents.base_agent import BaseAgent
from agents.moutet_maxime_submission01 import GreedyAgent
from utils.agent_utils import pi_uniform


@torch.no_grad()
def live_plot(data_dict, x_key=None, figsize=(7, 5), title=''):
    from matplotlib.ticker import MaxNLocator
    clear_output(wait=True)
    fig, ax1 = plt.subplots(figsize=figsize)

    x = data_dict[x_key] if x_key is not None else np.arange(len(next(iter(data_dict.values()))))

    # Courbes à afficher sur l'axe principal (y1)
    y1_keys = ['total_reward']
    y2_keys = [k for k in data_dict if k not in y1_keys + [x_key, 'success']]

    for label in y1_keys:
        if label in data_dict and len(data_dict[label]) > 0:
            ax1.plot(x, data_dict[label], label=label, linewidth=1)
    ax1.set_ylabel('Total Reward')
    ax1.grid(alpha=.5, which='both')

    # Axe secondaire (y2) pour n_steps, final_dist, etc.
    ax2 = ax1.twinx()
    for label in y2_keys:
        if len(data_dict[label]) > 0:
            ax2.plot(x, data_dict[label], label=label, linewidth=1, linestyle='--')
    ax2.set_ylabel('Other metrics')

    # Marquer les succès
    if "success" in data_dict:
        successes = np.array(data_dict["success"])
        x_success = np.array(x)[successes]
        y_success = np.array(data_dict["total_reward"])[successes]
        ax1.scatter(x_success, y_success, color='red', label='Success', zorder=5)

    # Fusionner les légendes des deux axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    ax1.set_xlabel('episode' if x_key else 'epoch')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


    

def distance_to_goal(s):
    """
    Calculate the distance to the goal.
    """
    goal = np.array([16, 31])
    return np.linalg.norm(s[:2] - goal)


def shaped_reward(s, s2, goal, stagnation_counter, step_count):
    prev_dist = distance_to_goal(s)
    new_dist = distance_to_goal(s2)
    delta = prev_dist - new_dist

    # Récompense pour le progrès
    progress_reward = delta

    # Pénalité pour la distance au but
    distance_penalty = -np.log(1 + new_dist)

    # Pénalité pour la stagnation
    stagnation_penalty = -10 if stagnation_counter > 5 else 0

    # Pénalité pour les longues épisodes
    episode_length_penalty = -0.1 * step_count

    # Récompense pour la réussite
    success_reward = 1000 * np.exp(-new_dist) if new_dist < 5 else 0

    # Récompense totale
    r = progress_reward + distance_penalty + stagnation_penalty + episode_length_penalty + success_reward

    return r



def collect_demo_trajectories(env, pi_demo, n_episodes=10):
    transitions = []
    goal = np.array([16, 31])
    for i in range(n_episodes):
        s, _ = env.reset()
        done = False
        stagnation_counter = 0
        step = 0
        while not done:
            a = pi_demo(s)
            s2, r, term, trunc, _ = env.step(a)
            step += 1
            r = shaped_reward(s, s2, goal, stagnation_counter, step)

            if distance_to_goal(s2) <= distance_to_goal(s):
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            transitions.append(s.tolist() +  [a, r, term] + s2.tolist())
            if term or trunc:
                done = True
                break
            s = s2
    return np.array(transitions)


def DQN(
        env,
        model,
        capacity=10000,
        batch_size=32,
        C=10,
        eps=0.1,
        gamma=0.99,
        n_iterations=10000,
        pi_demo=GreedyAgent(),
    ):
    Qmax = None
    s, _ = env.reset()
    n_actions = env.action_space.n
    pi = pi_uniform
    old_model = deepcopy(model)

    eps_start, decay_steps = 1,  n_iterations // 2
    eps_fn = lambda t: max(eps, eps_start - (eps_start - eps) * t / decay_steps)

    prob_start, prob_final = 1.0, 0.05
    demo_fn = lambda t: max(prob_final, prob_start - (prob_start - prob_final) * t / decay_steps)


    goal = np.array([16, 31])

    buffer = np.zeros((capacity, len(s) * 2 + 3))
    demo_transitions = collect_demo_trajectories(env, pi_demo, n_episodes=200)
    print(f"Demo successes: {np.sum([t[len(s) + 2] for t in demo_transitions]) / len(demo_transitions)}")
    for i, trans in enumerate(demo_transitions):
        if i >= capacity // 4:
            break
        buffer[i, :] = np.array(trans.tolist())

    pbar = tqdm(total=n_iterations)
    stagnation_counter = 0

    metrics = {
        "episode": [],
        "total_reward": [],
        "n_steps": [],
        "final_dist": [],
        "n_successes": [],
        "success": []
    }

    episode_idx = 0
    total_reward = 0
    step = 0
    n_successes = 0
    success = False
    for t in range(n_iterations):
        if np.random.rand() < eps_fn(t):
            a = pi_demo(s) if np.random.rand() < demo_fn(t) else pi_uniform(s)
        else:
            a = pi(s)
        s2, r, term, trunc, _ = env.step(a)
        step += 1

        r = shaped_reward(s, s2, goal, stagnation_counter, step)
        total_reward += r
        if distance_to_goal(s2) <= distance_to_goal(s):
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if distance_to_goal(s2) < 1.5:
            n_successes += 1
            success = True

        buffer[capacity // 4 + t %  (3 * capacity // 4), :] = np.array(s.copy().tolist() + [a, r, term] + s2.copy().tolist())
        if term or trunc:
            # s, _ = env.reset_close(variance=variance_fn(t))
            s, _ = env.reset()


            metrics["episode"].append(episode_idx)
            metrics["total_reward"].append(total_reward)
            metrics["n_steps"].append(step)
            metrics["final_dist"].append(distance_to_goal(s2))
            metrics["n_successes"].append(n_successes)
            metrics["success"].append(success)
            live_plot(metrics, x_key="episode", title="Training Progress")

            episode_idx += 1
            total_reward = 0
            success = False

            step = 0
            stagnation_counter = 0
        else:
            s = s2

        success_rate = np.sum(buffer[:, len(s) + 2]) / len(buffer)
        pbar.set_postfix(success_rate=success_rate)
        I = np.random.choice(min(t + 1, capacity), size=batch_size)
        data = buffer[I]

        if t % C == 0:
            old_model = deepcopy(model)

        Xb = data[:, :len(s) + 1]
        Yb = data[:, len(s) + 1]
        # Make a step
        if Qmax is None:
            model.partial_fit(Xb, Yb)

        Qmax = np.max(
            [
                old_model.predict(np.column_stack([
                            data[:, len(s) + 3:],
                            np.ones(len(data)).reshape(-1, 1) * a
                            ]))
                for a in range(n_actions)
            ],axis=0)
        Yb = data[:, len(s) + 1] + gamma * (1 - data[:, len(s) + 2]) * Qmax
        pbar.update(1)
        model.partial_fit(Xb, Yb)

        def pi(s):
            q = [model.predict(np.array(s.tolist() + [a]).reshape(1, -1))[0] for a in range(n_actions)]
            return np.argmax(q)
    return pi
