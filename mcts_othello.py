"""
=================================================================
  MCTS Othello — Avec affichage d'arbre, barre de progression
                 et sauvegarde/chargement du modèle
=================================================================

Nouveautés par rapport à la version de base :
  1. Barre de progression en console (entraînement et simulations)
  2. Affichage de l'arbre de décision avec poids UCB1 / wins / visits
  3. Table de valeurs persistante (sauvegarde JSON) :
       - Pendant l'entraînement : chaque état visité accumule des stats
       - Au chargement         : les stats précédentes sont réinjectées
                                 dans les nœuds MCTS (mémoire transitive)

  NOTE SUR LA SAUVEGARDE :
  ────────────────────────
  MCTS pur reconstruit son arbre à zéro à chaque décision — il n'y a
  pas de "poids" comme dans un réseau de neurones.
  Ce qu'on sauvegarde ici est une VALUE TABLE :
    clé   = hash de l'état du plateau (64 bytes → entier)
    valeur = {"visits": N, "wins": W}
  Au prochain lancement, ces statistiques sont réinjectées dans les
  nœuds dès qu'un état connu est rencontré. Le MCTS "se souvient"
  des positions qu'il a déjà évaluées.

Dépendances :
  pip install gymnasium numpy
=================================================================
"""

import math
import random
import time
import json
import hashlib
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────
BOARD_SIZE  = 8
EMPTY       = 0
BLACK       = 1
WHITE       = -1
DIRECTIONS  = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

MODEL_PATH  = "mcts_model.json"   # Fichier de sauvegarde


# ═════════════════════════════════════════════════════════════
#  UTILITAIRES CONSOLE
# ═════════════════════════════════════════════════════════════

def progress_bar(current: int, total: int, prefix: str = "", width: int = 40) -> str:
    """
    Génère une barre de progression ASCII.

    Exemple : Entraînement  [████████████░░░░░░░░]  60%  (6/10)
    """
    filled  = int(width * current / total)
    bar     = "█" * filled + "░" * (width - filled)
    percent = int(100 * current / total)
    return f"\r  {prefix}  [{bar}]  {percent:>3}%  ({current}/{total})"


def print_tree(node, depth: int = 0, max_depth: int = 2,
               max_children: int = 5, is_root: bool = True):
    """
    Affiche l'arbre de décision MCTS en console.

    Pour chaque nœud on affiche :
      Action   : case jouée (ligne, colonne) ou "PASSE"
      Visits   : nombre de fois que le nœud a été visité
      Wins     : score cumulé (victoires - défaites)
      W/V      : ratio victoires / visites  (exploitation)
      UCB1     : score UCB1 complet (si le nœud a un parent)

    Les enfants sont triés par nombre de visites décroissant.
    On n'affiche que les max_children premiers à chaque niveau
    pour garder l'affichage lisible.
    """
    if depth > max_depth:
        return

    indent  = "  " * depth
    branch  = "└─ " if depth > 0 else ""

    # Formater le nom de l'action
    if node.action is None:
        action_str = "RACINE"
    elif node.action == BOARD_SIZE * BOARD_SIZE:
        action_str = "PASSE "
    else:
        r, c = node.action // BOARD_SIZE, node.action % BOARD_SIZE
        action_str = f"({r},{c})  "

    # Ratio wins/visits
    ratio = node.wins / node.visits if node.visits > 0 else 0.0

    # Score UCB1 (seulement si le nœud a un parent et a été visité)
    if node.parent and node.parent.visits > 0 and node.visits > 0:
        ucb = node.ucb1(math.sqrt(2))
        ucb_str = f"UCB1={ucb:+.3f}"
    else:
        ucb_str = "UCB1=  n/a "

    # Barre visuelle proportionnelle aux visites
    max_v   = max((c.visits for c in node.parent.children), default=1) if node.parent else node.visits
    bar_len = int(20 * node.visits / max(max_v, 1))
    bar     = "▓" * bar_len + "░" * (20 - bar_len)

    print(f"{indent}{branch}"
          f"action={action_str}  "
          f"visits={node.visits:>5}  "
          f"wins={node.wins:>+7.1f}  "
          f"W/V={ratio:>+.3f}  "
          f"{ucb_str}  "
          f"|{bar}|")

    # Trier les enfants par visites décroissantes et n'afficher que les N premiers
    sorted_children = sorted(node.children, key=lambda n: n.visits, reverse=True)
    shown           = sorted_children[:max_children]
    hidden          = len(sorted_children) - len(shown)

    for child in shown:
        print_tree(child, depth + 1, max_depth, max_children, is_root=False)

    if hidden > 0:
        print(f"{'  ' * (depth+1)}└─ ... ({hidden} autre(s) non affichés)")


# ═════════════════════════════════════════════════════════════
#  ENVIRONNEMENT GYMNASIUM
# ═════════════════════════════════════════════════════════════
class OthelloEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode        = render_mode
        self.observation_space  = spaces.Box(low=-1, high=1,
                                             shape=(BOARD_SIZE, BOARD_SIZE),
                                             dtype=np.int8)
        self.action_space       = spaces.Discrete(BOARD_SIZE * BOARD_SIZE + 1)
        self.board              = None
        self.current_player     = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board              = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        mid                     = BOARD_SIZE // 2
        self.board[mid-1][mid-1] = WHITE
        self.board[mid-1][mid]   = BLACK
        self.board[mid][mid-1]   = BLACK
        self.board[mid][mid]     = WHITE
        self.current_player     = BLACK
        return self.board.copy(), {}

    def step(self, action):
        valid_moves = self.get_valid_moves(self.current_player)
        if action == BOARD_SIZE * BOARD_SIZE:
            if len(valid_moves) > 0:
                return self.board.copy(), -1.0, True, False, {"error": "passer sans raison"}
            self.current_player *= -1
            if len(self.get_valid_moves(self.current_player)) == 0:
                return self.board.copy(), self._final_reward(), True, False, {}
            return self.board.copy(), 0.0, False, False, {}
        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        if (row, col) not in valid_moves:
            return self.board.copy(), -1.0, True, False, {"error": "coup invalide"}
        self._apply_move(row, col, self.current_player)
        self.current_player *= -1
        if self._is_game_over():
            reward = self._final_reward(from_player=self.current_player * -1)
            return self.board.copy(), reward, True, False, {}
        if len(self.get_valid_moves(self.current_player)) == 0:
            self.current_player *= -1
        return self.board.copy(), 0.0, False, False, {}

    def render(self):
        symbols = {EMPTY: ".", BLACK: "●", WHITE: "○"}
        print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
        for r in range(BOARD_SIZE):
            print(f"{r} " + " ".join(symbols[self.board[r][c]] for c in range(BOARD_SIZE)))
        b = int(np.sum(self.board == BLACK))
        w = int(np.sum(self.board == WHITE))
        print(f"  Noirs(●): {b}  Blancs(○): {w}\n")

    def get_valid_moves(self, player):
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY and self._would_flip(r, c, player):
                    moves.append((r, c))
        return moves

    def _would_flip(self, row, col, player):
        opponent = -player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            found_opponent = False
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board[r][c] == opponent:
                    found_opponent = True
                elif self.board[r][c] == player and found_opponent:
                    return True
                else:
                    break
                r += dr; c += dc
        return False

    def _apply_move(self, row, col, player):
        opponent = -player
        self.board[row][col] = player
        for dr, dc in DIRECTIONS:
            r, c    = row + dr, col + dc
            to_flip = []
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board[r][c] == opponent:
                    to_flip.append((r, c))
                elif self.board[r][c] == player and to_flip:
                    for fr, fc in to_flip:
                        self.board[fr][fc] = player
                    break
                else:
                    break
                r += dr; c += dc

    def _is_game_over(self):
        return (len(self.get_valid_moves(BLACK)) == 0 and
                len(self.get_valid_moves(WHITE)) == 0)

    def _final_reward(self, from_player=BLACK):
        blacks = np.sum(self.board == BLACK)
        whites = np.sum(self.board == WHITE)
        if blacks > whites:   winner = BLACK
        elif whites > blacks: winner = WHITE
        else:                 return 0.0
        return 1.0 if winner == from_player else -1.0

    def clone(self):
        c               = OthelloEnv()
        c.board         = self.board.copy()
        c.current_player = self.current_player
        return c

    def board_hash(self) -> str:
        """
        Calcule un identifiant unique pour l'état courant du plateau.

        On convertit le tableau numpy en bytes puis on calcule un hash SHA-1
        tronqué. Deux plateaux identiques produisent toujours le même hash.
        C'est la clé utilisée dans la ValueTable pour retrouver les
        statistiques d'un état déjà vu.
        """
        return hashlib.sha1(self.board.tobytes()).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════
#  TABLE DE VALEURS — La "mémoire" du MCTS
# ═════════════════════════════════════════════════════════════
class ValueTable:
    """
    Stocke les statistiques visits/wins pour chaque état de plateau
    rencontré au fil des parties d'entraînement.

    Rôle dans MCTS :
    ────────────────
    MCTS standard oublie tout entre deux décisions. Avec une ValueTable,
    les nœuds héritent des statistiques accumulées sur toutes les parties
    précédentes dès qu'ils correspondent à un état déjà visité.
    Plus on joue de parties, plus la table est précise, et plus les
    décisions de MCTS s'améliorent sans recalculer depuis zéro.

    C'est le principe de la "mémoire transitive" utilisé dans AlphaGo.

    Format de sauvegarde (JSON) :
    {
      "metadata": {"total_games": 42, "total_updates": 15000, ...},
      "states":   {"hash1": {"visits": 120, "wins": 34.5}, ...}
    }
    """

    def __init__(self):
        # dict[hash → {"visits": int, "wins": float}]
        self.states: dict[str, dict] = {}
        self.total_games    = 0
        self.total_updates  = 0

    # ── Mise à jour d'un état ─────────────────────────────────
    def update(self, board_hash: str, result: float):
        if board_hash not in self.states:
            self.states[board_hash] = {"visits": 0, "wins": 0.0}
        self.states[board_hash]["visits"] += 1
        self.states[board_hash]["wins"]   += result
        self.total_updates += 1

    # ── Lecture d'un état ─────────────────────────────────────
    def get(self, board_hash: str) -> tuple[int, float]:
        """Retourne (visits, wins) ou (0, 0.0) si inconnu."""
        if board_hash in self.states:
            s = self.states[board_hash]
            return s["visits"], s["wins"]
        return 0, 0.0

    # ── Sauvegarde JSON ───────────────────────────────────────
    def save(self, path: str = MODEL_PATH):
        """
        Sérialise la table dans un fichier JSON.
        On sauvegarde aussi des métadonnées pour afficher les infos
        de la dernière session au rechargement.
        """
        data = {
            "metadata": {
                "total_games"  : self.total_games,
                "total_updates": self.total_updates,
                "known_states" : len(self.states),
                "saved_at"     : time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "states": self.states,
        }
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        size_kb = os.path.getsize(path) / 1024
        print(f"\n  ✔ Modèle sauvegardé → {path}  "
              f"({len(self.states)} états, {size_kb:.1f} Ko)")

    # ── Chargement JSON ───────────────────────────────────────
    def load(self, path: str = MODEL_PATH) -> bool:
        """
        Charge la table depuis un fichier JSON.
        Retourne True si le chargement a réussi, False sinon.
        """
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            data = json.load(f)
        meta             = data.get("metadata", {})
        self.states      = data.get("states", {})
        self.total_games = meta.get("total_games", 0)
        self.total_updates = meta.get("total_updates", 0)
        print(f"  ✔ Modèle chargé ← {path}")
        print(f"    {len(self.states)} états connus  |  "
              f"{self.total_games} parties d'entraînement  |  "
              f"sauvegardé le {meta.get('saved_at', '?')}")
        return True


# ═════════════════════════════════════════════════════════════
#  NŒUD MCTS — avec injection de la ValueTable
# ═════════════════════════════════════════════════════════════
class MCTSNode:
    """
    Nœud de l'arbre MCTS.

    Nouveauté par rapport à la version de base :
    Si une ValueTable est fournie et que l'état courant est connu,
    on initialise visits et wins avec les statistiques mémorisées.
    Le nœud "démarre chaud" au lieu de démarrer à zéro — les
    décisions sont meilleures dès la première simulation.
    """

    def __init__(self, state: OthelloEnv, parent=None, action=None,
                 value_table: ValueTable = None):
        self.state        = state
        self.parent       = parent
        self.action       = action
        self.children     = []
        self.value_table  = value_table

        # Initialisation depuis la ValueTable si disponible
        if value_table is not None:
            h                = state.board_hash()
            visits, wins     = value_table.get(h)
            self.visits      = visits
            self.wins        = wins
        else:
            self.visits = 0
            self.wins   = 0.0

        valid = state.get_valid_moves(state.current_player)
        if valid:
            self.untried_actions = [r * BOARD_SIZE + c for r, c in valid]
        else:
            self.untried_actions = [BOARD_SIZE * BOARD_SIZE]

    def ucb1(self, C: float) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits +
                C * math.sqrt(math.log(self.parent.visits) / self.visits))

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.state._is_game_over()


# ═════════════════════════════════════════════════════════════
#  MCTS — avec ValueTable, barre de progression, affichage arbre
# ═════════════════════════════════════════════════════════════
class MCTS:
    """
    Monte Carlo Tree Search avec :
      - Injection d'une ValueTable pour la mémoire transitive
      - Barre de progression optionnelle pendant les simulations
      - Affichage de l'arbre de décision après chaque coup
    """

    def __init__(self,
                 num_simulations     : int   = 500,
                 exploration_constant: float = math.sqrt(2),
                 value_table         : ValueTable = None,
                 show_progress       : bool  = False,
                 show_tree           : bool  = False,
                 tree_depth          : int   = 2,
                 tree_children       : int   = 5):
        self.num_simulations       = num_simulations
        self.exploration_constant  = exploration_constant
        self.value_table           = value_table
        self.show_progress         = show_progress
        self.show_tree             = show_tree
        self.tree_depth            = tree_depth
        self.tree_children         = tree_children

    def choose_action(self, env: OthelloEnv) -> int:
        """
        Lance num_simulations itérations MCTS et retourne le meilleur coup.
        Affiche une barre de progression si show_progress=True.
        Affiche l'arbre si show_tree=True.
        """
        root = MCTSNode(state=env.clone(), value_table=self.value_table)

        for i in range(self.num_simulations):

            # Barre de progression (mise à jour tous les 10 pas pour la vitesse)
            if self.show_progress and i % 10 == 0:
                print(progress_bar(i, self.num_simulations,
                                   prefix="  Simulations"), end="", flush=True)

            # ① Sélection
            node = self._select(root)
            # ② Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            # ③ Simulation
            result = self._simulate(node)
            # ④ Rétropropagation
            self._backpropagate(node, result)

        if self.show_progress:
            print(progress_bar(self.num_simulations, self.num_simulations,
                               prefix="  Simulations"))
            print()  # Nouvelle ligne après la barre

        # ── Affichage de l'arbre ──────────────────────────────
        if self.show_tree and root.children:
            player_name = "Noirs (●)" if env.current_player == BLACK else "Blancs (○)"
            print(f"\n  ┌─ ARBRE DE DÉCISION — {player_name} "
                  f"({self.num_simulations} simulations) ─────────────")
            print(f"  │  Profondeur affichée : {self.tree_depth}  "
                  f"│  Top {self.tree_children} enfants par nœud")
            print(f"  │  W/V = wins/visits  │  UCB1 = score de sélection")
            print(f"  └{'─'*60}")
            print_tree(root,
                       max_depth   = self.tree_depth,
                       max_children= self.tree_children)
            print()

        # Choisir l'enfant le plus visité
        if not root.children:
            return BOARD_SIZE * BOARD_SIZE
        best = max(root.children, key=lambda n: n.visits)
        return best.action

    def _select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = max(node.children,
                       key=lambda n: n.ucb1(self.exploration_constant))
        return node

    def _expand(self, node):
        action  = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        new_env = node.state.clone()
        new_env.step(action)
        child   = MCTSNode(state=new_env, parent=node, action=action,
                           value_table=self.value_table)
        node.children.append(child)
        return child

    def _simulate(self, node):
        sim_env        = node.state.clone()
        player_at_root = sim_env.current_player
        while not sim_env._is_game_over():
            valid  = sim_env.get_valid_moves(sim_env.current_player)
            action = (random.choice(valid[0:1] + valid) if valid
                      else BOARD_SIZE * BOARD_SIZE)
            if valid:
                r, c   = random.choice(valid)
                action = r * BOARD_SIZE + c
            else:
                action = BOARD_SIZE * BOARD_SIZE
            _, _, done, _, _ = sim_env.step(action)
            if done:
                break
        return sim_env._final_reward(from_player=player_at_root)

    def _backpropagate(self, node, result):
        """
        Remonte le résultat dans l'arbre ET met à jour la ValueTable
        pour que chaque état visité soit mémorisé.
        """
        while node is not None:
            node.visits += 1
            node.wins   += result
            # Mettre à jour la mémoire persistante
            if self.value_table is not None:
                h = node.state.board_hash()
                self.value_table.update(h, result)
            result = -result
            node   = node.parent


# ═════════════════════════════════════════════════════════════
#  SESSION D'ENTRAÎNEMENT — avec barre de progression globale
# ═════════════════════════════════════════════════════════════
class TrainingSession:
    """
    Entraîne le MCTS en le faisant jouer contre lui-même (self-play)
    et accumule les statistiques dans une ValueTable.

    Self-play :
    ───────────
    MCTS joue les deux couleurs. Chaque partie enrichit la ValueTable
    avec des statistiques sur tous les états traversés. Les parties
    suivantes démarrent avec une meilleure connaissance des positions.

    À la fin, la ValueTable est sauvegardée sur disque.
    """

    def __init__(self,
                 n_games    : int = 20,
                 mcts_sims  : int = 200,
                 model_path : str = MODEL_PATH):
        self.n_games    = n_games
        self.mcts_sims  = mcts_sims
        self.model_path = model_path
        self.value_table = ValueTable()

        # Tenter de charger un modèle existant
        print("\n" + "═" * 60)
        print("  INITIALISATION")
        if self.value_table.load(model_path):
            print(f"  Les statistiques précédentes seront réutilisées.\n")
        else:
            print(f"  Aucun modèle trouvé — entraînement depuis zéro.\n")

    def run(self):
        """Lance la session d'entraînement avec self-play."""
        agent = MCTS(
            num_simulations      = self.mcts_sims,
            value_table          = self.value_table,
            show_progress        = False,  # Désactivé pendant l'entraînement
            show_tree            = False,  #   (on utilise la barre globale)
        )

        print("═" * 60)
        print(f"  ENTRAÎNEMENT — Self-play")
        print(f"  {self.n_games} parties  ×  {self.mcts_sims} simulations/coup\n")

        wins_black = wins_white = draws = 0

        for i in range(self.n_games):
            # Barre de progression globale
            print(progress_bar(i, self.n_games,
                               prefix="  Parties  "), end="", flush=True)

            # Jouer une partie complète
            env        = OthelloEnv()
            env.reset()
            terminated = False

            while not terminated:
                valid = env.get_valid_moves(env.current_player)
                if not valid:
                    _, _, terminated, _, _ = env.step(BOARD_SIZE * BOARD_SIZE)
                    continue
                action = agent.choose_action(env)
                _, _, terminated, _, _ = env.step(action)

            # Résultat
            b = int(np.sum(env.board == BLACK))
            w = int(np.sum(env.board == WHITE))
            if b > w:   wins_black += 1
            elif w > b: wins_white += 1
            else:       draws      += 1

            self.value_table.total_games += 1

        # Barre finale à 100%
        print(progress_bar(self.n_games, self.n_games, prefix="  Parties  "))
        print()

        # Statistiques
        print("─" * 60)
        print(f"  RÉSULTATS DE L'ENTRAÎNEMENT")
        print(f"  Victoires Noirs  : {wins_black:>4}")
        print(f"  Victoires Blancs : {wins_white:>4}")
        print(f"  Nuls             : {draws:>4}")
        print(f"  États mémorisés  : {len(self.value_table.states):>6}")
        print(f"  Total mises à jour : {self.value_table.total_updates:>8}")

        # Sauvegarde
        self.value_table.save(self.model_path)
        print("═" * 60)

        return self.value_table


# ═════════════════════════════════════════════════════════════
#  PARTIE AVEC AFFICHAGE COMPLET
# ═════════════════════════════════════════════════════════════
def play_game_display(value_table: ValueTable = None,
                      mcts_sims: int = 300,
                      show_tree: bool = True):
    """
    Joue une partie MCTS vs MCTS avec :
      - Barre de progression des simulations à chaque coup
      - Affichage de l'arbre de décision
      - Réutilisation de la ValueTable si disponible
    """
    env   = OthelloEnv()
    env.reset()

    agent = MCTS(
        num_simulations      = mcts_sims,
        value_table          = value_table,
        show_progress        = True,
        show_tree            = show_tree,
        tree_depth           = 2,
        tree_children        = 5,
    )

    print("\n" + "═" * 60)
    vt_info = (f"{len(value_table.states)} états mémorisés"
               if value_table else "sans mémoire")
    print(f"  PARTIE  —  MCTS ({mcts_sims} sims, {vt_info})")
    print("═" * 60)
    env.render()

    terminated = False
    move_count = 0

    while not terminated:
        player      = env.current_player
        valid_moves = env.get_valid_moves(player)
        name        = "Noirs (●)" if player == BLACK else "Blancs (○)"

        if not valid_moves:
            print(f"  {name} passe son tour.\n")
            _, _, terminated, _, _ = env.step(BOARD_SIZE * BOARD_SIZE)
            continue

        move_count += 1
        print(f"  ── Tour {move_count} — {name} réfléchit... ──")
        action = agent.choose_action(env)
        r, c   = action // BOARD_SIZE, action % BOARD_SIZE
        print(f"  Coup choisi : ({r}, {c})\n")

        _, _, terminated, _, _ = env.step(action)
        env.render()

    # Résultat
    b = int(np.sum(env.board == BLACK))
    w = int(np.sum(env.board == WHITE))
    print("═" * 60)
    print(f"  FIN — Noirs: {b}  Blancs: {w}")
    if b > w:   print("  🏆 Victoire des NOIRS !")
    elif w > b: print("  🏆 Victoire des BLANCS !")
    else:       print("  🤝 Égalité !")
    print("═" * 60)


# ═════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("""
  ╔══════════════════════════════════════════════════════╗
  ║   MCTS Othello — Avec mémoire et affichage           ║
  ╠══════════════════════════════════════════════════════╣
  ║  1 → Entraînement (self-play) + sauvegarde           ║
  ║  2 → Partie avec arbre de décision                   ║
  ║      (charge le modèle sauvegardé si disponible)     ║
  ║  3 → Entraînement PUIS partie                        ║
  ╚══════════════════════════════════════════════════════╝
    """)

    choix = input("  Votre choix (1/2/3) : ").strip()

    if choix == "1":
        n      = int(input("  Nombre de parties d'entraînement [défaut 20] : ") or "20")
        sims   = int(input("  Simulations par coup              [défaut 200] : ") or "200")
        session = TrainingSession(n_games=n, mcts_sims=sims)
        session.run()

    elif choix == "2":
        sims = int(input("  Simulations par coup [défaut 300] : ") or "300")
        tree = input("  Afficher l'arbre de décision ? (o/n) [défaut o] : ").strip().lower()
        show_tree = (tree != "n")
        # Charger le modèle s'il existe
        vt = ValueTable()
        loaded = vt.load(MODEL_PATH)
        if not loaded:
            print("  (Aucun modèle trouvé — partie sans mémoire)")
            vt = None
        play_game_display(value_table=vt, mcts_sims=sims, show_tree=show_tree)

    elif choix == "3":
        n_train = int(input("  Parties d'entraînement [défaut 10] : ") or "10")
        sims    = int(input("  Simulations par coup   [défaut 200] : ") or "200")
        session = TrainingSession(n_games=n_train, mcts_sims=sims)
        vt      = session.run()
        print("\n  Lancement d'une partie avec le modèle entraîné...")
        play_game_display(value_table=vt, mcts_sims=sims, show_tree=True)

    else:
        print("  Choix invalide — lancement du mode 2 par défaut.")
        vt = ValueTable()
        if not vt.load(MODEL_PATH):
            vt = None
        play_game_display(value_table=vt, mcts_sims=300, show_tree=True)