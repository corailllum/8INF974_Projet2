import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import os

#définition des paramètres
EPISODES      = 1000
MAX_STEPS     = 1000
BATCH_SIZE    = 32
GAMMA         = 0.99
LR            = 1e-4
MEMORY_SIZE   = 10_000
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.97        #décroissance de l'epsilon par épisode
TARGET_UPDATE = 10
FRAME_STACK   = 4
FRAME_SIZE    = (84, 84)

#définition des récompenses
REWARD_WIN      =  10.0
REWARD_LOSE     = -10.0
REWARD_CAPTURE  =   1.0  
REWARD_CAPTURED =  -1.0   
REWARD_ILLEGAL  =  -2.0   

EMPTY      = 0
BLACK      = 1   #notre joueur
WHITE      = -1  #l'adversaire
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def init_board():
    #initialise le plateau othello standard avec les 4 pions du centre
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    mid = BOARD_SIZE // 2
    board[mid-1][mid-1] = WHITE
    board[mid-1][mid]   = BLACK
    board[mid][mid-1]   = BLACK
    board[mid][mid]     = WHITE
    return board

def would_flip(board, row, col, player):
    #vérifie si jouer en (row, col) retourne au moins un pion adverse
    opponent = -player
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        found_opponent = False
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board[r][c] == opponent:
                found_opponent = True
            elif board[r][c] == player and found_opponent:
                return True  #on encadre des pions adverses donc coup légal
            else:
                break
            r += dr; c += dc
    return False

def get_valid_moves(board, player):
    #retourne la liste des indices (0-63) des coups légaux pour ce joueur
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == EMPTY and would_flip(board, r, c, player):
                moves.append(r * BOARD_SIZE + c)
    return moves

def apply_move(board, row, col, player):
    #applique un coup sur le plateau et retourne les pions capturés
    opponent = -player
    board[row][col] = player
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board[r][c] == opponent:
                to_flip.append((r, c))
            elif board[r][c] == player and to_flip:
                for fr, fc in to_flip:
                    board[fr][fc] = player  #on retourne les pions capturés
                break
            else:
                break
            r += dr; c += dc
    return board

#preprocessing : conversion rgb vers niveaux de gris
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

class FrameStack:
    #création d'une mémoire de 4 frames pour l'agent pour déduire le mouvement des pions
    def __init__(self, n):
        self.n = n
        self.frames = deque(maxlen=n)  #file pour garder les n dernières frames

    def reset(self, frame):
        processed = preprocess_frame(frame)  #on preprocess (rgb2gris)
        for _ in range(self.n):
            self.frames.append(processed)    #on remplit le début de la pile
        return self._get_state()

    def step(self, frame):
        self.frames.append(preprocess_frame(frame))  #on ajoute la nouvelle frame, la plus ancienne sort
        return self._get_state()

    def _get_state(self):
        return np.stack(self.frames, axis=0)  #(4, 84, 84)

#réseau de neurones avec 64 sorties, une par case du plateau
class DQN(nn.Module):
    def __init__(self, n_actions=64):
        super(DQN, self).__init__()
        #analyse de l'image en 3 passes conv2d pour récupérer les détails importants
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),  #distinction du plateau
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            #distinctions des pions
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            #position exacte des pions
            nn.ReLU()
        )
        conv_out = self._get_conv_output((FRAME_STACK, *FRAME_SIZE))  #mesure la taille de sortie pour la couche suivante
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),    #condense les valeurs
            nn.ReLU(),                   #nettoyage
            nn.Linear(512, n_actions)    #réduction à 64 valeurs, une par case du plateau
        )

    def _get_conv_output(self, shape):  #sert à mesurer la taille de sortie des convolutions
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):  #visualisation puis décisionnel puis retour des scores
        x = self.conv(x)
        x = x.view(x.size(0), -1)  #on aplatit pour passer aux couches linéaires
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):  #création de la file de mémoire
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):  #ajoute un souvenir dans la file
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  #pioche aléatoirement dans la file pour entraîner sans repartir de 0
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (  #on transforme en tensor pytorch avant d'envoyer
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

#définition de l'agent avec détection automatique du meilleur processeur disponible
class DQNAgent:
    def __init__(self):
        self.n_actions = 64  #une sortie par case du plateau
        self.epsilon   = EPSILON_START

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")    #apple silicon
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")   #gpu nvidia
        else:
            self.device = torch.device("cpu")    #cpu fallback
        print(f" Utilisation du device : {self.device}")

        self.policy_net = DQN(self.n_actions).to(self.device)  #réseau principal qui apprend
        self.target_net = DQN(self.n_actions).to(self.device)  #réseau de référence stable
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  #le réseau cible n'apprend pas, il sert juste de référence

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  #optimiseur adam
        self.memory    = ReplayBuffer(MEMORY_SIZE)  #mémoire de l'agent
        self.steps     = 0  #réservé pour des améliorations futures

    def select_action(self, state, valid_moves):
        #choisit une action uniquement parmi les coups légaux
        #si valid_moves est vide on retourne none pour passer le tour
        if not valid_moves:
            return None

        if random.random() < self.epsilon:  #exploration uniquement parmi les coups légaux
            return random.choice(valid_moves)

        #exploitation : on masque les cases illégales avec -inf
        #argmax ne peut donc jamais choisir une case interdite
        with torch.no_grad():
            state_t  = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0)
            mask = torch.full((self.n_actions,), float('-inf')).to(self.device)
            for move in valid_moves:
                mask[move] = q_values[move]  #on remet la vraie valeur q sur les cases légales
            return mask.argmax().item()

    def store(self, *args):
        self.memory.push(*args)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None  #pas assez de souvenirs donc on ne fait rien

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  #on ne récupère que les actions réellement faites
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]  #meilleure récompense future estimée
            target = rewards + GAMMA * next_q * (1 - dones)  #récompense cible

        loss = nn.SmoothL1Loss()(q_values, target)  #calcul de l'erreur
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        #copie les poids du réseau principal dans le réseau cible tous les n épisodes
        self.target_net.load_state_dict(self.policy_net.state_dict())

#boucle d'entraînement
def train():
    #on crée l'env atari, on initialise le dqn et on crée la pile de frames
    env     = gym.make("ALE/Othello-v5", render_mode=None)
    agent   = DQNAgent()
    stacker = FrameStack(FRAME_STACK)

    print(f"  Entraînement DQN sur Othello Atari")
    print(f"  Épisodes : {EPISODES}")

    episode_rewards = []

    for episode in range(1, EPISODES + 1):
        obs, info = env.reset()  #on réinitialise le jeu
        #on envoie plusieurs fire pour être sûr du démarrage de la partie
        for _ in range(60):
            obs, _, _, _, info = env.step(1)  #fire
        for _ in range(20):
            obs, _, _, _, info = env.step(0)  #noop pour laisser le jeu s'initialiser
        state = stacker.reset(obs)  #on initialise la pile avec la première observation

        total_reward   = 0.0
        board          = init_board()  #plateau miroir initialisé comme le jeu
        current_player = BLACK

        for step in range(MAX_STEPS):
            valid_moves    = get_valid_moves(board, current_player)
            opponent_moves = get_valid_moves(board, -current_player)

            #fin de partie si aucun des deux joueurs ne peut jouer
            if not valid_moves and not opponent_moves:
                black_count = int(np.sum(board == BLACK))
                white_count = int(np.sum(board == WHITE))
                if black_count > white_count:
                    total_reward += REWARD_WIN
                elif black_count < white_count:
                    total_reward += REWARD_LOSE
                break

            #choix de l'action parmi les coups légaux uniquement
            action_idx = agent.select_action(state, valid_moves)

            if action_idx is None:
                #pas de coup possible, on passe son tour avec noop
                obs, _, terminated, truncated, _ = env.step(0)
                current_player = -current_player
                reward = 0.0
            else:
                #on compte les pions de noir avant le coup quelle que soit la couleur qui joue
                noir_avant = int(np.sum(board == BLACK))
                #on envoie noop à atari car la synchro curseur ne fonctionne pas
                obs, _, terminated, truncated, _ = env.step(0)
                target_row = action_idx // BOARD_SIZE
                target_col = action_idx % BOARD_SIZE
                board      = apply_move(board, target_row, target_col, current_player)
                #on compte les pions de noir après le coup
                noir_apres = int(np.sum(board == BLACK))

                if current_player == BLACK:
                    #noir a joué : récompense pour les captures
                    captures = noir_apres - noir_avant - 1  #-1 car on a posé un pion
                    reward   = REWARD_CAPTURE * captures if captures > 0 else 0.0
                else:
                    #blanc a joué : pénalité pour les pions de noir perdus
                    pertes = noir_avant - noir_apres
                    reward = REWARD_CAPTURED * pertes if pertes > 0 else 0.0

                current_player = -current_player

            done = terminated or truncated
            next_state = stacker.step(obs)  #on ajoute la frame à la pile

            if action_idx is not None:
                agent.store(state, action_idx, reward, next_state, float(done))  #sauvegarde dans la mémoire
                agent.learn()

            state        = next_state
            total_reward += reward

            if done:
                break

        #résultat final de la partie calculé avec le plateau copie
        black_count = int(np.sum(board == BLACK))
        white_count = int(np.sum(board == WHITE))
        if black_count > white_count:
            resultat = "NOIR gagne "
        elif white_count > black_count:
            resultat = "BLANC gagne"
        else:
            resultat = "Egalite    "

        episode_rewards.append(total_reward)  #sauvegarde de la récompense de l'épisode
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)  #diminution de l'epsilon

        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        #chaque épisode s'affiche
        avg = np.mean(episode_rewards[-10:])
        print(f"Épisode {episode:4d}/{EPISODES} | "
              f"Récompense : {total_reward:8.1f} | "
              f"Moy   : {avg:8.1f} | "
              f"Epsilon   : {agent.epsilon:.3f} | "
              f"{resultat} ({black_count}-{white_count})")

    env.close()
    print("\nEntraînement terminé ")
    torch.save(agent.policy_net.state_dict(), "dqn_othello.pth")
    print(" Modèle sauvegardé : dqn_othello.pth")

    return agent

#démo finale avec affichage pygame et enregistrement vidéo
def demo(agent, video_path="othello_demo.mp4"):
    print(f"\nLancement de la démo finale")
    import pygame

    #on utilise env atari pour les observations mais affichage d une copie du plateau en pygame
    #car la problème synchro curseur environnement
    env     = gym.make("ALE/Othello-v5", render_mode=None)
    stacker = FrameStack(FRAME_STACK)

    obs, info = env.reset()
    state     = stacker.reset(obs)

    #démarrage de la partie
    for _ in range(60):
        obs, _, _, _, _ = env.step(1)  #fire
        state = stacker.step(obs)
    for _ in range(20):
        obs, _, _, _, _ = env.step(0)  #noop
        state = stacker.step(obs)

    agent.epsilon  = 0.0  #pas d'exploration en démo
    board          = init_board()
    current_player = BLACK

    #constantes d'affichage pygame
    CELL_SIZE   = 70
    MARGIN      = 40
    WIN_SIZE    = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
    COLOR_BG    = (20,  120,  40)   
    COLOR_GRID  = (0,   80,   0)    
    COLOR_BLACK = (20,  20,   20)  
    COLOR_WHITE = (230, 230,  230) 
    COLOR_HINT  = (100, 200, 100)   
    COLOR_TEXT  = (255, 255,  255)  

    pygame.init()
    screen = pygame.display.set_mode((WIN_SIZE, WIN_SIZE + 60))
    pygame.display.set_caption("DQN Othello - Démonstration")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 22, bold=True)

    frames_video = []  #liste des frames pour la vidéo

    def draw_board(board, current_player, valid_moves, step, score_black, score_white):
        #dessin du plateau avec pygame puis génération d'une frame cv2 pour la vidéo
        screen.fill(COLOR_BG)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                pygame.draw.rect(screen, COLOR_GRID, (x, y, CELL_SIZE, CELL_SIZE), 1)
                action_idx = r * BOARD_SIZE + c
                if action_idx in valid_moves:
                    #point vert indique cases jouables
                    pygame.draw.circle(screen, COLOR_HINT,
                                       (x + CELL_SIZE//2, y + CELL_SIZE//2), 8)
                if board[r][c] == BLACK:
                    pygame.draw.circle(screen, COLOR_BLACK,
                                       (x + CELL_SIZE//2, y + CELL_SIZE//2),
                                       CELL_SIZE//2 - 6)
                elif board[r][c] == WHITE:
                    pygame.draw.circle(screen, COLOR_WHITE,
                                       (x + CELL_SIZE//2, y + CELL_SIZE//2),
                                       CELL_SIZE//2 - 6)
        player_name = "NOIR" if current_player == BLACK else "BLANC"
        info_text = font.render(
            f"Tour {step:3d} | {player_name} joue | Noir: {score_black}  Blanc: {score_white}",
            True, COLOR_TEXT
        )
        screen.blit(info_text, (MARGIN, WIN_SIZE + 15))
        pygame.display.flip()

        #frame vidéo générée cv2
        frame_cv2 = np.zeros((WIN_SIZE + 60, WIN_SIZE, 3), dtype=np.uint8)
        frame_cv2[:, :] = COLOR_BG[::-1]  #bgr pour cv2
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                cv2.rectangle(frame_cv2, (x, y), (x + CELL_SIZE, y + CELL_SIZE),
                               COLOR_GRID[::-1], 1)
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                action_idx = r * BOARD_SIZE + c
                if action_idx in valid_moves:
                    cv2.circle(frame_cv2, (cx, cy), 8, COLOR_HINT[::-1], -1)
                if board[r][c] == BLACK:
                    cv2.circle(frame_cv2, (cx, cy), CELL_SIZE // 2 - 6, COLOR_BLACK[::-1], -1)
                elif board[r][c] == WHITE:
                    cv2.circle(frame_cv2, (cx, cy), CELL_SIZE // 2 - 6, COLOR_WHITE[::-1], -1)
        label = f"Tour {step:3d} | {player_name} joue | Noir:{score_black} Blanc:{score_white}"
        cv2.putText(frame_cv2, label, (MARGIN, WIN_SIZE + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame_cv2

    running = True
    step    = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        valid_moves    = get_valid_moves(board, current_player)
        opponent_moves = get_valid_moves(board, -current_player)
        score_black    = int(np.sum(board == BLACK))
        score_white    = int(np.sum(board == WHITE))

        #dessin + capture de la frame
        frame = draw_board(board, current_player, valid_moves, step, score_black, score_white)
        frames_video.append(frame)

        #fin de partie si prsn ne put jouer
        if not valid_moves and not opponent_moves:
            running = False
            break

        #choix du coup par l'agent
        action_idx = agent.select_action(state, valid_moves)

        if action_idx is None:
            obs, _, _, _, _ = env.step(0)  #noop si aucun coup possible
            current_player  = -current_player
        else:
            obs, _, _, _, _ = env.step(0)  #noop  atari
            target_row = action_idx // BOARD_SIZE
            target_col = action_idx % BOARD_SIZE
            board      = apply_move(board, target_row, target_col, current_player)
            current_player = -current_player

        state = stacker.step(obs)
        step += 1
        clock.tick(2)  #2 cps/s

    #résultat final
    score_black = int(np.sum(board == BLACK))
    score_white = int(np.sum(board == WHITE))
    result = "NOIR gagne !" if score_black > score_white else \
             "BLANC gagne !" if score_white > score_black else "Égalité !"
    print(f"\n Résultat : {result}  (Noir: {score_black} | Blanc: {score_white})")
    pygame.time.wait(2000)
    pygame.quit()
    env.close()

    #sauvegarde de la vidéo déjà  bgr -> besoin de convertir
    if frames_video:
        h, w   = frames_video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(video_path, fourcc, 2.0, (w, h))
        for f in frames_video:
            out.write(f)
        out.release()
        print(f" Vidéo sauvegardée : {video_path} ({len(frames_video)} frames)")

#main
if __name__ == "__main__":
    trained_agent = train()
    demo(trained_agent)
