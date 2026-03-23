import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy



#  CONSTANTES DU JEU

BOARD_SIZE  = 8          # Plateau 8×8
EMPTY       = 0          # Case vide
BLACK       = 1          # Joueur 1 (noir)
WHITE       = -1         # Joueur 2 (blanc)

# Les 8 directions possibles sur le plateau
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


#  1. ENVIRONNEMENT GYMNASIUM – OthelloEnv

class OthelloEnv(gym.Env):
    """
    Environnement Gymnasium pour le jeu Othello.

    Observation :  tableau numpy (8,8) avec les valeurs : -1 (blanc), 0 (vide), 1 (noir)
    Action      :  entier [0, 64]  où 64 = passer son tour
                   sinon action = row*8 + col
    Reward      :  +1 si victoire, -1 si defaite, 0 si nul ou en cours
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Espace d'observations : plateau 8×8 avec valeurs dans {-1, 0, 1}
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8
        )
        # Espace d'actions : 64 cases + 1 action "passer"
        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE + 1)

        self.board        = None   # L'etat du plateau
        self.current_player = None # Joueur actuel : BLACK ou WHITE

    # Reinitialise l'environnement
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Plateau vide
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        # Position initiale d'Othello : 4 pions au centre
        mid = BOARD_SIZE // 2
        self.board[mid-1][mid-1] = WHITE
        self.board[mid-1][mid]   = BLACK
        self.board[mid][mid-1]   = BLACK
        self.board[mid][mid]     = WHITE

        self.current_player = BLACK  # Les noirs commencent
        return self.board.copy(), {}

    # Applique une action et renvoie le nouvel etat
    def step(self, action):
       
        valid_moves = self.get_valid_moves(self.current_player)

        # Action "passer" 
        if action == BOARD_SIZE * BOARD_SIZE:
            # Verifier que passer est justifie (aucun coup valide)
            if len(valid_moves) > 0:
                # Coup invalide : on penalise et on termine
                return self.board.copy(), -1.0, True, False, {"error": "passer sans raison"}
            # Changer de joueur
            self.current_player *= -1
            # Si l'autre joueur n'a pas non plus de coup = fin de partie
            if len(self.get_valid_moves(self.current_player)) == 0:
                return self.board.copy(), self._final_reward(), True, False, {}
            return self.board.copy(), 0.0, False, False, {}

        # Coup normal
        row, col = action // BOARD_SIZE, action % BOARD_SIZE

        # Verifier la validite du coup
        if (row, col) not in valid_moves:
            return self.board.copy(), -1.0, True, False, {"error": "coup invalide"}

        # Appliquer le coup
        self._apply_move(row, col, self.current_player)

        # Changer de joueur
        self.current_player *= -1

        # Verifier si la partie est terminee
        if self._is_game_over():
            reward = self._final_reward(from_player=self.current_player * -1)
            return self.board.copy(), reward, True, False, {}

        # Si le nouveau joueur n'a aucun coup = il passe automatiquement
        if len(self.get_valid_moves(self.current_player)) == 0:
            self.current_player *= -1

        return self.board.copy(), 0.0, False, False, {}

    # Rendu textuel du plateau
    def render(self):
        symbols = {EMPTY: ".", BLACK: "●", WHITE: "○"}
        print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
        for r in range(BOARD_SIZE):
            row_str = f"{r} " + " ".join(symbols[self.board[r][c]] for c in range(BOARD_SIZE))
            print(row_str)
        blacks = np.sum(self.board == BLACK)
        whites = np.sum(self.board == WHITE)
        print(f"  Noirs(●): {blacks}  Blancs(○): {whites}")
        print()

    # Retourne la liste des coups valides pour un joueur 
    def get_valid_moves(self, player):
        """Retourne une liste de tuples (row, col) des cases jouables."""
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY and self._would_flip(r, c, player):
                    moves.append((r, c))
        return moves

    # Verifie si au moins un pion serait retourne 
    def _would_flip(self, row, col, player):
        """Retourne True si jouer en (row,col) retourne au moins un pion adverse."""
        opponent = -player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            found_opponent = False
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board[r][c] == opponent:
                    found_opponent = True
                elif self.board[r][c] == player and found_opponent:
                    return True  # On encadre des pions adverses
                else:
                    break
                r += dr; c += dc
        return False

    # ── Applique un coup sur le plateau ───────────────────────
    def _apply_move(self, row, col, player):
        """Place le pion et retourne tous les pions captures."""
        opponent = -player
        self.board[row][col] = player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board[r][c] == opponent:
                    to_flip.append((r, c))
                elif self.board[r][c] == player and to_flip:
                    for fr, fc in to_flip:
                        self.board[fr][fc] = player  # Retourner les pions
                    break
                else:
                    break
                r += dr; c += dc

    # ── Verifie si la partie est terminee ─────────────────────
    def _is_game_over(self):
        """La partie se termine quand aucun des deux joueurs ne peut jouer"""
        return (
            len(self.get_valid_moves(BLACK)) == 0 and
            len(self.get_valid_moves(WHITE)) == 0
        )

    # ── Calcule la recompense finale ──────────────────────────
    def _final_reward(self, from_player=BLACK):
        blacks = np.sum(self.board == BLACK)
        whites = np.sum(self.board == WHITE)
        if blacks > whites:
            winner = BLACK
        elif whites > blacks:
            winner = WHITE
        else:
            return 0.0  # Nul

        return 1.0 if winner == from_player else -1.0

    # ── Clone l'environnement (utile pour MCTS) ───────────────
    def clone(self):
        """Retourne une copie independante de l'environnement"""
        cloned          = OthelloEnv()
        cloned.board    = self.board.copy()
        cloned.current_player = self.current_player
        return cloned


class MCTSNode:
    """
    Represente un nœud dans l'arbre Monte Carlo.

    Attributs cles :
      state Copie de l'environnement à ce nœud
      parent Nœud parent (None pour la racine)
      action Action qui a mene à ce nœud depuis le parent
      children Liste des nœuds fils dejà explores
      untried_actions Actions non encore developpees
      visits Nombre de fois que ce nœud a ete visite
      wins Nombre de victoires accumulees (du point de vue du joueur QUI A JOUe pour arriver ici)
    """

    def __init__(self, state: OthelloEnv, parent=None, action=None):
        self.state   = state          # etat courant (copie de l'env)
        self.parent  = parent         # Nœud parent
        self.action  = action         # Action prise depuis le parent

        self.children        = []     # Enfants developpes
        self.visits          = 0      # Nombre de visites
        self.wins            = 0.0    # Score cumule

        # Actions non encore explorees depuis ce nœud
        valid = state.get_valid_moves(state.current_player)
        if valid:
            self.untried_actions = [r * BOARD_SIZE + c for r, c in valid]
        else:
            self.untried_actions = [BOARD_SIZE * BOARD_SIZE]  # Passer

    # UCB1
    def ucb1(self, exploration_constant: float) -> float:
        """
        Upper Confidence Bound (UCB1) :
          UCB1 = (wins / visits) + C * sqrt(ln(parent.visits) / visits)

          - wins/visits      : exploitation  (preferer les nœuds gagnants)
          - sqrt(ln(N)/n)    : exploration   (preferer les nœuds peu visites)
          - C                : constante d'equilibre (souvent sqrt(2))
        """
        if self.visits == 0:
            return float("inf")  # Priorite absolue aux nœuds non visites
        exploitation = self.wins / self.visits
        exploration  = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    #Noeud totalement developpe
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    # Noeud terminal  
    def is_terminal(self) -> bool:
        return self.state._is_game_over()

    def __repr__(self):
        return f"MCTSNode(visits={self.visits}, wins={self.wins:.1f}, action={self.action})"


class MCTS:
    """
    Paramètres :
      num_simulations    – Nombre d'iterations MCTS par decision
      exploration_constant – Constante C de UCB1 (defaut : sqrt(2))
    """

    def __init__(self, num_simulations: int = 500, exploration_constant: float = math.sqrt(2)):
        self.num_simulations      = num_simulations
        self.exploration_constant = exploration_constant

    # Choisit le meilleur coup pour l'etat courant 
    def choose_action(self, env: OthelloEnv) -> int:
        """
        Lance `num_simulations` iterations MCTS depuis l'etat `env`
        et retourne l'action (int) estimee comme la meilleure.
        """
        root = MCTSNode(state=env.clone())

        for _ in range(self.num_simulations):
            node = self._select(root)

            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)

            result = self._simulate(node)

            self._backpropagate(node, result)

        # Choisir l'enfant le plus visite
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action

    #   selection
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Descend dans l'arbre en selectionnant à chaque niveau
        le fils avec le meilleur score UCB1.

        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node  # Ce nœud necessite une expansion
            # Selectionner l'enfant de meilleur UCB1
            node = max(
                node.children,
                key=lambda n: n.ucb1(self.exploration_constant)
            )
        return node  # Nœud terminal
    #   Expension
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Choisit une action non encore exploree (au hasard),
        l'applique à une copie de l'etat, et ajoute le nœud fils.
        """
        # Choisir une action non exploree aleatoirement
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Appliquer l'action dans une copie de l'environnement
        new_env = node.state.clone()
        new_env.step(action)

        # Creer et enregistrer le nouveau nœud fils
        child = MCTSNode(state=new_env, parent=node, action=action)
        node.children.append(child)
        return child

    # Rollout
    def _simulate(self, node: MCTSNode) -> float:
        """
        Joue aleatoirement jusqu'à la fin de la partie.
        Retourne le resultat du point de vue du joueur de la RACINE.

        """
        sim_env        = node.state.clone()
        player_at_root = sim_env.current_player  # Joueur qui veut maximiser

        while not sim_env._is_game_over():
            valid = sim_env.get_valid_moves(sim_env.current_player)
            if valid:
                # random rollout
                r, c    = random.choice(valid)
                action  = r * BOARD_SIZE + c
            else:
                action  = BOARD_SIZE * BOARD_SIZE  # Passer

            _, reward, done, _, _ = sim_env.step(action)
            if done:
                break

        # evaluer le resultat du point de vue du joueur à la racine
        return sim_env._final_reward(from_player=player_at_root)

   
    #   Retropropagation
    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Remonte le resultat depuis le nœud feuille jusqu'à la racine
        """
        while node is not None:
            node.visits += 1
            node.wins   += result
            result       = -result  # L'adversaire voit le resultat inverse
            node         = node.parent


def play_game(
    mcts_black: MCTS,
    mcts_white: MCTS,
    render: bool = True
) -> int:
    """
    Fait jouer une partie complète entre deux agents MCTS.

    Paramètres :
      mcts_black  – Agent MCTS pour les noirs (peut être None pour joueur humain)
      mcts_white  – Agent MCTS pour les blancs
      render      – Afficher le plateau à chaque coup

    """
    env = OthelloEnv(render_mode="human")
    obs, _ = env.reset()

    if render:
        print("═" * 30)
        print("  DeBUT DE LA PARTIE")
        print("═" * 30)
        env.render()

    terminated = False
    move_count = 0

    while not terminated:
        player = env.current_player
        agent  = mcts_black if player == BLACK else mcts_white
        player_name = "Noirs (●)" if player == BLACK else "Blancs (○)"

        valid_moves = env.get_valid_moves(player)

        # Aucun coup possible → passer automatiquement
        if not valid_moves:
            print(f"  {player_name} passe son tour.")
            obs, reward, terminated, _, info = env.step(BOARD_SIZE * BOARD_SIZE)
            continue

        # Choisir le coup
        action = agent.choose_action(env)
        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        move_count += 1

        if render:
            print(f"  Tour {move_count:>2} — {player_name} joue en ({row}, {col})")

        obs, reward, terminated, _, info = env.step(action)

        if render:
            env.render()

    # ── Resultat final ────────────────────────────────────────
    blacks = int(np.sum(env.board == BLACK))
    whites = int(np.sum(env.board == WHITE))

    print("═" * 30)
    print(f"  FIN DE PARTIE  —  Noirs: {blacks}  Blancs: {whites}")

    if blacks > whites:
        print("   Victoire des NOIRS !")
        return BLACK
    elif whites > blacks:
        print("   Victoire des BLANCS !")
        return WHITE
    else:
        print("   egalite !")
        return 0


if __name__ == "__main__":
    print("  MCTS Othello — Demonstration")

    # ── Mode 1 : Une partie MCTS vs MCTS ──────────────────────
    print("\n[Mode 1] Une partie MCTS(300) vs MCTS(100)\n")
    agent_noir  = MCTS(num_simulations=300, exploration_constant=math.sqrt(2))
    agent_blanc = MCTS(num_simulations=100, exploration_constant=math.sqrt(2))
    play_game(agent_noir, agent_blanc, render=True)
