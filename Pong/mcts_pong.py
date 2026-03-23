import math
import random
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
 
# NŒUD DE L'ARBRE MCTS 

class MCTSNode:
    def __init__(self, state, parent=None, action=None,
                 untried_actions=None, done=False):
        self.state           = state
        self.parent          = parent
        self.action          = action
        self.children        = {}
        self.visits          = 0
        self.value           = 0.0
        self.untried_actions = untried_actions if untried_actions else []
        self.done            = done

    def ucb1(self, c: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration  = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = math.sqrt(2)) -> "MCTSNode":
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def most_visited_child(self) -> "MCTSNode":
        return max(self.children.values(), key=lambda n: n.visits)

# WRAPPER D'ENVIRONNEMENT AVEC SNAPSHOT

class SnapshotEnv:
    """
    Encapsule un environnement Gymnasium/ALE.
    .unwrapped.ale donne accès direct à l'émulateur Atari
    pour cloneState / restoreState.
    """

    def __init__(self, env_id: str = "ALE/Pong-v5", render: bool = False):
        render_mode    = "human" if render else None
        self.env       = gym.make(env_id, render_mode=render_mode)
        self.ale       = self.env.unwrapped.ale
        self.n_actions = self.env.action_space.n

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def save_state(self):
        return self.ale.cloneState()

    def restore_state(self, state_token):
        self.ale.restoreState(state_token)

    def close(self):
        self.env.close()

# ALGORITHME MCTS

class MCTS:
    """
    Utilise env_search (sans rendu) pour toutes les simulations.
    L'environnement d'affichage est géré séparément dans la boucle principale.
    """

    def __init__(self, env_search: SnapshotEnv, n_simulations=50,
                 rollout_depth=30, c_ucb=math.sqrt(2), gamma=0.99):
        self.env           = env_search
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.c_ucb         = c_ucb
        self.gamma         = gamma

    def search(self, root_state_token) -> int:
        self.env.restore_state(root_state_token)
        root = MCTSNode(
            state           = root_state_token,
            untried_actions = list(range(self.env.n_actions)),
        )
        for _ in range(self.n_simulations):
            self.env.restore_state(root_state_token)
            node   = self._select(root)
            if not node.done and not node.is_fully_expanded():
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        return root.most_visited_child().action

    def _select(self, node):
        while not node.done and node.is_fully_expanded():
            node = node.best_child(self.c_ucb)
            _, _, done, _ = self.env.step(node.action)
            node.done = done
        return node

    def _expand(self, node):
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        _, _, done, _ = self.env.step(action)
        child_state   = self.env.save_state()
        child = MCTSNode(
            state           = child_state,
            parent          = node,
            action          = action,
            untried_actions = list(range(self.env.n_actions)),
            done            = done,
        )
        node.children[action] = child
        return child

    def _simulate(self, node):
        if node.done:
            return 0.0
        total_reward, discount = 0.0, 1.0
        for _ in range(self.rollout_depth):
            action = random.randint(0, self.env.n_actions - 1)
            _, reward, done, _ = self.env.step(action)
            total_reward += discount * reward
            discount     *= self.gamma
            if done:
                break
        return total_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value  += reward
            node         = node.parent

# BOUCLE PRINCIPALE — DOUBLE ENVIRONNEMENT (AFFICHAGE + SIMULATION)

def run_mcts_pong(n_episodes=2, n_simulations=30,
                  rollout_depth=20):
    """
    Deux environnements en parallèle :
      env_search  — sans rendu, dédié aux simulations MCTS (rapide)
      env_display — avec rendu human, affiche le vrai déroulement du jeu

    Après chaque décision MCTS, on applique la même action dans env_display
    pour garder les deux environnements synchronisés.
    """

    print("Initialisation des environnements...")
    env_search  = SnapshotEnv("ALE/Pong-v5", render=False)  # simulations
    env_display = SnapshotEnv("ALE/Pong-v5", render=True)   # affichage

    mcts = MCTS(env_search, n_simulations=n_simulations,
                rollout_depth=rollout_depth)

    for episode in range(n_episodes):
        # Réinitialiser les deux environnements
        env_search.reset()
        env_display.reset()

        # Synchroniser env_search sur l'état initial de env_display
        # (les seeds peuvent différer, on force la synchro via state copy)
        init_state = env_display.save_state()
        env_search.restore_state(init_state)

        total_reward, step, done = 0.0, 0, False

        print(f"\n{''*40}")
        print(f"  Épisode {episode + 1} / {n_episodes}")
        print(f"{''*40}")

        while not done:
            #  Snapshot depuis env_display (état de référence) 
            root_state = env_display.save_state()

            #  MCTS cherche la meilleure action (sur env_search) 
            action = mcts.search(root_state)

            #  Appliquer l'action réelle sur env_display (avec rendu) 
            _, reward, done, _ = env_display.step(action)
            total_reward += reward
            step         += 1

            #  Synchroniser env_search sur le nouvel état réel 
            new_state = env_display.save_state()
            env_search.restore_state(new_state)

            if step % 50 == 0:
                print(f"  Step {step:4d} | Action {action} | "
                      f"Récompense cumulée : {total_reward:.1f}")

        print(f"\n  ✓ Fin épisode {episode + 1} — "
              f"{step} steps — Récompense totale : {total_reward:.1f}")

    env_search.close()
    env_display.close()
    print("\nTerminé.")


# MAIN

if __name__ == "__main__":
    run_mcts_pong(
        n_episodes    = 2,
        n_simulations = 30,
        rollout_depth = 20,
    )