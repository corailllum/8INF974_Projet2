import gymnasium as gym
import ale_py
import numpy as np
import pickle
import os
import time

gym.register_envs(ale_py)

def save_tree(root, filename="mcts_tree.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(root, f)

def load_tree(filename="mcts_tree.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

class MCTSNode:
    def __init__(self, state, env, parent=None, action=None):
        self.state = state
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = list(range(env.action_space.n))

    def best_child(self, c=1.41):
        for child in self.children:
            if child.visits == 0:
                return child

        def ucb(child):
            exploit = child.wins / child.visits
            explore = c * np.sqrt(np.log(self.visits) / child.visits)
            return exploit + explore

        return max(self.children, key=ucb)

    def expand(self):
        action = self.untried_actions.pop()

        self.env.unwrapped.ale.restoreState(self.state)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        new_state = self.env.unwrapped.ale.cloneState()

        child = MCTSNode(new_state, self.env, parent=self, action=action)
        self.children.append(child)

        return child, reward, done

    def rollout(self):
        done = False
        total_reward = 0.0

        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

        return total_reward

    def backpropagate(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

def mcts(env, simulations=100, root=None):
    if root is None:
        root = MCTSNode(env.unwrapped.ale.cloneState(), env)

    for i in range(simulations):
        if i % 10 == 0:
            print(f"   simulation {i}/{simulations}")

        node = root
        env.unwrapped.ale.restoreState(root.state)

        # Selection
        while node.untried_actions == [] and node.children:
            node = node.best_child()
            obs, reward, terminated, truncated, _ = env.step(node.action)
            done = terminated or truncated

        # Expansion
        if node.untried_actions:
            node, reward, done = node.expand()

        # Simulation
        if not done:
            reward = node.rollout()

        # Backprop
        node.backpropagate(reward)

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action, root


def select_action_from_tree(root):
    if root is None or not root.children:
        return None
    return max(root.children, key=lambda c: c.visits).action

# Train = entrainement de l'arbre sur plusieurs parties 

def train(num_games=50, simulations=200):
    print(" Début de l'entraînement...")
    env = gym.make("ALE/Othello-v5")  # pas de rendu

    root = load_tree()
    start_time = time.time()

    for game in range(num_games):
        print(f" Game {game+1} en cours...")
        game_start = time.time()

        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            action, root = mcts(env, simulations, root)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        # Stats
        elapsed = time.time() - start_time
        avg_time = elapsed / (game + 1)
        remaining = avg_time * (num_games - game - 1)

        print(f"[{game+1}/{num_games}] "
              f"Steps={steps} | "
              f"Avg={avg_time:.2f}s | "
              f"ETA={remaining/60:.2f} min")

    save_tree(root)
    env.close()

# Evaluation = jouer une partie avec l'arbre de train sans faire d'entrainement supplémentaire

def play_game_evaluation():
    env = gym.make("ALE/Othello-v5", render_mode='human')

    root = load_tree()
    if root is None:
        print(" Aucun arbre trouvé. Lance train() d'abord.")
        return

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action_from_tree(root)

        # fallback sécurité
        if action is None:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"[EVAL] Game over! Total reward: {total_reward}")
    env.close()


if __name__ == '__main__':
    #  Entraînement 
    train(num_games=10, simulations=50)

    #  Test de l'arbre 
    play_game_evaluation()