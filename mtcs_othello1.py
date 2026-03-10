"""
  Monte Carlo Tree Search (MCTS) 

"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy


# ─────────────────────────────────────────────────────────────
#  CONSTANTES DU JEU
# ─────────────────────────────────────────────────────────────
BOARD_SIZE  = 8          # Plateau 8×8
EMPTY       = 0          # Case vide
BLACK       = 1          # Joueur 1 (noir)
WHITE       = -1         # Joueur 2 (blanc)

# Les 8 directions possibles sur le plateau
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


# ═════════════════════════════════════════════════════════════
#  1. ENVIRONNEMENT GYMNASIUM – OthelloEnv
# ═════════════════════════════════════════════════════════════
class OthelloEnv(gym.Env):
    """
    Environnement Gymnasium pour le jeu Othello.

    Observation :  tableau numpy (8,8) avec les valeurs : -1 (blanc), 0 (vide), 1 (noir)
    Action      :  entier [0, 64]  où 64 = passer son tour
                   sinon action = row*8 + col
    Reward      :  +1 si victoire, -1 si défaite, 0 si nul ou en cours
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

        self.board        = None   # L'état du plateau
        self.current_player = None # Joueur actuel : BLACK ou WHITE

    # ── Réinitialise l'environnement ──────────────────────────
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

    # ── Applique une action et renvoie le nouvel état ─────────
    def step(self, action):
        """
        action : int [0..64]
          - 0..63 : jouer à la case (action//8, action%8)
          - 64    : passer son tour
        """
        valid_moves = self.get_valid_moves(self.current_player)

        # --- Action "passer" ---
        if action == BOARD_SIZE * BOARD_SIZE:
            # Vérifier que passer est justifié (aucun coup valide)
            if len(valid_moves) > 0:
                # Coup invalide : on pénalise et on termine
                return self.board.copy(), -1.0, True, False, {"error": "passer sans raison"}
            # Changer de joueur
            self.current_player *= -1
            # Si l'autre joueur n'a pas non plus de coup → fin de partie
            if len(self.get_valid_moves(self.current_player)) == 0:
                return self.board.copy(), self._final_reward(), True, False, {}
            return self.board.copy(), 0.0, False, False, {}

        # --- Coup normal ---
        row, col = action // BOARD_SIZE, action % BOARD_SIZE

        # Vérifier la validité du coup
        if (row, col) not in valid_moves:
            return self.board.copy(), -1.0, True, False, {"error": "coup invalide"}

        # Appliquer le coup
        self._apply_move(row, col, self.current_player)

        # Changer de joueur
        self.current_player *= -1

        # Vérifier si la partie est terminée
        if self._is_game_over():
            reward = self._final_reward(from_player=self.current_player * -1)
            return self.board.copy(), reward, True, False, {}

        # Si le nouveau joueur n'a aucun coup → il passe automatiquement
        if len(self.get_valid_moves(self.current_player)) == 0:
            self.current_player *= -1

        return self.board.copy(), 0.0, False, False, {}

    # ── Rendu textuel du plateau ──────────────────────────────
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

    # ── Retourne la liste des coups valides pour un joueur ────
    def get_valid_moves(self, player):
        """Retourne une liste de tuples (row, col) des cases jouables."""
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY and self._would_flip(r, c, player):
                    moves.append((r, c))
        return moves

    # ── Vérifie si au moins un pion serait retourné ───────────
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
        """Place le pion et retourne tous les pions capturés."""
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

    # ── Vérifie si la partie est terminée ─────────────────────
    def _is_game_over(self):
        """La partie se termine quand aucun des deux joueurs ne peut jouer."""
        return (
            len(self.get_valid_moves(BLACK)) == 0 and
            len(self.get_valid_moves(WHITE)) == 0
        )

    # ── Calcule la récompense finale ──────────────────────────
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
        """Retourne une copie indépendante de l'environnement."""
        cloned          = OthelloEnv()
        cloned.board    = self.board.copy()
        cloned.current_player = self.current_player
        return cloned


# ═════════════════════════════════════════════════════════════
#  2. NŒUD DE L'ARBRE MCTS – MCTSNode
# ═════════════════════════════════════════════════════════════
class MCTSNode:
    """
    Représente un nœud dans l'arbre Monte Carlo.

    Attributs clés :
      state       – Copie de l'environnement à ce nœud
      parent      – Nœud parent (None pour la racine)
      action      – Action qui a mené à ce nœud depuis le parent
      children    – Liste des nœuds fils déjà explorés
      untried_actions – Actions non encore développées
      visits      – Nombre de fois que ce nœud a été visité
      wins        – Nombre de victoires accumulées (du point de vue du joueur QUI A JOUÉ pour arriver ici)
    """

    def __init__(self, state: OthelloEnv, parent=None, action=None):
        self.state   = state          # État courant (copie de l'env)
        self.parent  = parent         # Nœud parent
        self.action  = action         # Action prise depuis le parent

        self.children        = []     # Enfants développés
        self.visits          = 0      # Nombre de visites
        self.wins            = 0.0    # Score cumulé

        # Actions non encore explorées depuis ce nœud
        valid = state.get_valid_moves(state.current_player)
        if valid:
            self.untried_actions = [r * BOARD_SIZE + c for r, c in valid]
        else:
            self.untried_actions = [BOARD_SIZE * BOARD_SIZE]  # Passer

    # ── UCB1 : formule de sélection ───────────────────────────
    def ucb1(self, exploration_constant: float) -> float:
        """
        Upper Confidence Bound (UCB1) :
          UCB1 = (wins / visits) + C * sqrt(ln(parent.visits) / visits)

          - wins/visits      : exploitation  (préférer les nœuds gagnants)
          - sqrt(ln(N)/n)    : exploration   (préférer les nœuds peu visités)
          - C                : constante d'équilibre (souvent sqrt(2) ≈ 1.414)
        """
        if self.visits == 0:
            return float("inf")  # Priorité absolue aux nœuds non visités
        exploitation = self.wins / self.visits
        exploration  = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    # ── Nœud totalement développé ? ──────────────────────────
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    # ── Nœud terminal (fin de partie) ? ──────────────────────
    def is_terminal(self) -> bool:
        return self.state._is_game_over()

    def __repr__(self):
        return f"MCTSNode(visits={self.visits}, wins={self.wins:.1f}, action={self.action})"


# ═════════════════════════════════════════════════════════════
#  3. ALGORITHME MCTS
# ═════════════════════════════════════════════════════════════
class MCTS:
    """
    Monte Carlo Tree Search — 4 phases :

      ① SÉLECTION    : Descendre dans l'arbre en choisissant l'enfant de meilleur UCB1
      ② EXPANSION    : Ajouter un nouveau nœud enfant non encore exploré
      ③ SIMULATION   : Jouer aléatoirement jusqu'à la fin de la partie (rollout)
      ④ RÉTROPROPAGATION : Remonter le résultat dans tous les ancêtres

    Paramètres :
      num_simulations    – Nombre d'itérations MCTS par décision
      exploration_constant – Constante C de UCB1 (défaut : sqrt(2))
    """

    def __init__(self, num_simulations: int = 500, exploration_constant: float = math.sqrt(2)):
        self.num_simulations      = num_simulations
        self.exploration_constant = exploration_constant

    # ── Choisit le meilleur coup pour l'état courant ──────────
    def choose_action(self, env: OthelloEnv) -> int:
        """
        Lance `num_simulations` itérations MCTS depuis l'état `env`
        et retourne l'action (int) estimée comme la meilleure.
        """
        root = MCTSNode(state=env.clone())

        for _ in range(self.num_simulations):
            # ① Sélection
            node = self._select(root)

            # ② Expansion (si le nœud n'est pas terminal)
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)

            # ③ Simulation (rollout)
            result = self._simulate(node)

            # ④ Rétropropagation
            self._backpropagate(node, result)

        # Choisir l'enfant le plus visité (stratégie robuste)
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action

    # ─────────────────────────────────────────────────────────
    #  ① SÉLECTION
    # ─────────────────────────────────────────────────────────
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Descend dans l'arbre en sélectionnant à chaque niveau
        le fils avec le meilleur score UCB1.

        On s'arrête quand :
          - le nœud est terminal (fin de partie), ou
          - le nœud n'est pas encore totalement développé
            (il reste des actions non explorées → expansion)
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node  # Ce nœud nécessite une expansion
            # Sélectionner l'enfant de meilleur UCB1
            node = max(
                node.children,
                key=lambda n: n.ucb1(self.exploration_constant)
            )
        return node  # Nœud terminal

    # ─────────────────────────────────────────────────────────
    #  ② EXPANSION
    # ─────────────────────────────────────────────────────────
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Choisit une action non encore explorée (au hasard),
        l'applique à une copie de l'état, et ajoute le nœud fils.
        """
        # Choisir une action non explorée aléatoirement
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Appliquer l'action dans une copie de l'environnement
        new_env = node.state.clone()
        new_env.step(action)

        # Créer et enregistrer le nouveau nœud fils
        child = MCTSNode(state=new_env, parent=node, action=action)
        node.children.append(child)
        return child

    # ─────────────────────────────────────────────────────────
    #  ③ SIMULATION (Rollout)
    # ─────────────────────────────────────────────────────────
    def _simulate(self, node: MCTSNode) -> float:
        """
        Joue aléatoirement jusqu'à la fin de la partie.
        Retourne le résultat du point de vue du joueur de la RACINE.

        Note : On clone l'état pour ne pas modifier l'arbre.
        """
        sim_env        = node.state.clone()
        player_at_root = sim_env.current_player  # Joueur qui veut maximiser

        while not sim_env._is_game_over():
            valid = sim_env.get_valid_moves(sim_env.current_player)
            if valid:
                # Politique aléatoire (random rollout)
                r, c    = random.choice(valid)
                action  = r * BOARD_SIZE + c
            else:
                action  = BOARD_SIZE * BOARD_SIZE  # Passer

            _, reward, done, _, _ = sim_env.step(action)
            if done:
                break

        # Évaluer le résultat du point de vue du joueur à la racine
        return sim_env._final_reward(from_player=player_at_root)

    # ─────────────────────────────────────────────────────────
    #  ④ RÉTROPROPAGATION
    # ─────────────────────────────────────────────────────────
    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Remonte le résultat depuis le nœud feuille jusqu'à la racine.

        Chaque nœud incrémente :
          - visits  (toujours +1)
          - wins    (résultat du point de vue du joueur du nœud parent,
                     car c'est le parent qui a CHOISI cette action)

        On alterne le signe du résultat à chaque niveau car les
        joueurs alternent (jeu à somme nulle).
        """
        while node is not None:
            node.visits += 1
            node.wins   += result
            result       = -result  # L'adversaire voit le résultat inversé
            node         = node.parent


# ═════════════════════════════════════════════════════════════
#  4. BOUCLE DE JEU PRINCIPALE
# ═════════════════════════════════════════════════════════════
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

    Retourne : BLACK (1), WHITE (-1), ou 0 (nul)
    """
    env = OthelloEnv(render_mode="human")
    obs, _ = env.reset()

    if render:
        print("═" * 30)
        print("  DÉBUT DE LA PARTIE")
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

    # ── Résultat final ────────────────────────────────────────
    blacks = int(np.sum(env.board == BLACK))
    whites = int(np.sum(env.board == WHITE))

    print("═" * 30)
    print(f"  FIN DE PARTIE  —  Noirs: {blacks}  Blancs: {whites}")

    if blacks > whites:
        print("  🏆 Victoire des NOIRS !")
        return BLACK
    elif whites > blacks:
        print("  🏆 Victoire des BLANCS !")
        return WHITE
    else:
        print("  🤝 Égalité !")
        return 0


# ═════════════════════════════════════════════════════════════
#  5. TOURNOI – Évaluation statistique
# ═════════════════════════════════════════════════════════════
def tournament(n_games: int = 10, sims_a: int = 200, sims_b: int = 50):
    """
    Joue `n_games` parties entre deux agents MCTS de forces différentes.
    Utile pour valider que plus de simulations = meilleur agent.

    Paramètres :
      sims_a  – Nombre de simulations pour l'agent A (plus fort)
      sims_b  – Nombre de simulations pour l'agent B (plus faible)
    """
    agent_a = MCTS(num_simulations=sims_a)
    agent_b = MCTS(num_simulations=sims_b)

    wins_a, wins_b, draws = 0, 0, 0

    print(f"\n{'═'*40}")
    print(f"  TOURNOI : Agent A ({sims_a} sims) vs Agent B ({sims_b} sims)")
    print(f"  {n_games} parties\n")

    for i in range(n_games):
        # Alterner qui commence pour éviter le biais du premier joueur
        if i % 2 == 0:
            result = play_game(agent_a, agent_b, render=False)
            if   result == BLACK:  wins_a += 1
            elif result == WHITE:  wins_b += 1
            else: draws += 1
        else:
            result = play_game(agent_b, agent_a, render=False)
            if   result == WHITE:  wins_a += 1  # A joue blanc cette fois
            elif result == BLACK:  wins_b += 1
            else: draws += 1

        print(f"  Partie {i+1:>2} terminée  |  A={wins_a}  B={wins_b}  Nul={draws}")

    print(f"\n  Résultats finaux sur {n_games} parties :")
    print(f"  Agent A ({sims_a} sims) : {wins_a} victoires ({100*wins_a//n_games}%)")
    print(f"  Agent B ({sims_b} sims) : {wins_b} victoires ({100*wins_b//n_games}%)")
    print(f"  Égalités              : {draws}")
    print("═" * 40)


# ═════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  MCTS Othello — Démonstration")
    print("═" * 50)

    # ── Mode 1 : Une partie MCTS vs MCTS ──────────────────────
    print("\n[Mode 1] Une partie MCTS(300) vs MCTS(100)\n")
    agent_noir  = MCTS(num_simulations=300, exploration_constant=math.sqrt(2))
    agent_blanc = MCTS(num_simulations=100, exploration_constant=math.sqrt(2))
    play_game(agent_noir, agent_blanc, render=True)

    # ── Mode 2 : Tournoi statistique (désactiver pour aller vite) ──
    # print("\n[Mode 2] Tournoi statistique\n")
    # tournament(n_games=20, sims_a=200, sims_b=50)