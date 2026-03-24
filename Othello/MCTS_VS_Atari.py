import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

# Fonctions utiles
def get_valid_actions(env):
    legal = env.unwrapped.ale.getMinimalActionSet()
    if legal is None or len(legal) == 0:
        return list(range(env.action_space.n))
    return list(legal)

# Node
class MCTSNode:
    def __init__(self, state, env, parent=None, action=None):
        self.state = state
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = get_valid_actions(env)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c=1.41):
        def ucb(child):
            if child.visits == 0:
                return float("inf")
            exploit = child.wins / child.visits
            explore = c * np.sqrt(np.log(self.visits) / child.visits) # self = child's parent
            return exploit + explore

        return max(self.children, key=ucb)

    def expand(self):
        action = self.untried_actions.pop()

        self.env.unwrapped.ale.restoreState(self.state) # Récupère la dernière sauvegarde du jeu

        obs, reward, terminated, truncated, inf = self.env.step(action)
        done = terminated or truncated

        new_state = self.env.unwrapped.ale.cloneState() # Sauvegarde de la partie

        child = MCTSNode(new_state, self.env, parent=self, action=action)
        self.children.append(child)

        return child, reward, done


    def rollout(self):
        self.env.unwrapped.ale.restoreState(self.state)

        done = False
        total_reward = 0.0

        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

        return total_reward

    def backpropagate(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

# MCTS
def mcts(env, simulations=200):
    root_state = env.unwrapped.ale.cloneState()
    root = MCTSNode(root_state, env) # Sauvegarde l’état actuel du jeu pour créer la racine.

    for s in range(simulations):
        node = root

        env.unwrapped.ale.restoreState(root.state) # Se remettre à la racine de l'arbre pour une nouvelle simulation

        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            _, _, terminated, truncated, info = env.step(node.action)
            done = terminated or truncated

        # 2. Expansion
        if not node.is_fully_expanded():
            node, reward, done = node.expand()
        else:
            done = False

        # 3. Simulation
        if not done:
            reward = node.rollout()

        # 4. Backpropagation
        node.backpropagate(reward)

        print("Simulation: ", s, "reward: ", reward)


    best_child = max(root.children, key=lambda c: c.visits)
    print("Best child Action: ", best_child.action)
    return best_child.action

# Play
def play_game(simulations=200):
    env = gym.make("ALE/Othello-v5", render_mode='human')

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = mcts(env, simulations)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        print(total_reward, " reward for action", action)

    print(f"Game over! Total reward: {total_reward}")
    env.close()

def evaluate(n_games=10):
    env = gym.make("ALE/Othello-v5")

    wins = 0
    draws = 0
    losses = 0

    for i in range(n_games):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = mcts(env, simulations=200)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1

        print(f"Game {i+1}: reward={total_reward}")

    print("\n=== RESULTS ===")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win rate: {wins / n_games:.2f}")

    env.close()

if __name__ == '__main__':
    #play_game(10)
    evaluate(10)
