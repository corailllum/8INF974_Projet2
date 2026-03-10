import gymnasium as gym
import ale_py
import numpy as np
# import gymcts :  A Monte Carlo Tree Search Library for Gymnasium-style Environments
gym.register_envs(ale_py)

"""TODO:
-   remplacer new_state par valeur du bot
-   remplacer ou enlever clone_state et restor_state
-   trouver des exemples de mcts avec gymnasium
-   trouver un moyen de limiter les actions de l agent a des actions possibles dans le jeu (arreter de tester toutes les case)
"""

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
            explore = c * np.sqrt(np.log(self.visits) / child.visits) # self = child's parent
            return exploit + explore

        return max(self.children, key=ucb)

    def expand(self):
        action = self.untried_actions.pop()

        self.env.unwrapped.ale.restoreState(self.state) # Récupère la dernière sauvegarde du jeu
        # je peux pas utiliser obs comme new-state car ce n est pas l etat complet du jeu : pas reproductible
        obs, reward, terminated, truncated, inf = self.env.step(action)
        done = terminated or truncated

        new_state = self.env.unwrapped.ale.cloneState() # Sauvegarde de la partie : score + plateau du jeu + tour de joueur

        child = MCTSNode(new_state, self.env, parent=self, action=action)
        self.children.append(child)

        return child, reward, done


    def rollout(self):
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

def mcts(env, simulations=100):
    root = MCTSNode(env.unwrapped.ale.cloneState(), env) # Sauvegarde l’état actuel du jeu pour créer la racine.

    for _ in range(simulations):
        node = root
        env.unwrapped.ale.restoreState(root.state) # Se remettre À la racine de l'arbre pour une nouvelle simulation

        # 1. Selection
        while node.untried_actions == [] and node.children:
            node = node.best_child()
            obs, reward, terminated, truncated, info = env.step(node.action)
            done = terminated or truncated

        # 2. Expansion
        if node.untried_actions:
            node, reward, done = node.expand()

        # 3. Simulation
        if not done:
            reward = node.rollout()

        # 4. Backpropagation
        node.backpropagate(reward)

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action

def play_game():
    env = gym.make("ALE/Othello-v5", render_mode='human')

    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = mcts(env, 500)
        obs, reward, terminated, truncated, info = env.step(action)
        print(terminated)
        done = terminated or truncated
        total_reward += reward

    print(f"Game over! Total reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    play_game()