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
EPISODES         = 1000        
MAX_STEPS        = 1000        
BATCH_SIZE       = 32
GAMMA            = 0.99        
LR               = 1e-4       
MEMORY_SIZE      = 10_000      
EPSILON_START    = 1.0        
EPSILON_END      = 0.05        
EPSILON_DECAY    = 0.97       
TARGET_UPDATE    = 10          
FRAME_STACK      = 4           
FRAME_SIZE       = (84, 84)    

#dfinition des récompenss
REWARD_WIN            =  10.0
REWARD_LOSE           = -10.0
REWARD_CAPTURE        =   1.0  
REWARD_CAPTURED       =  -1.0   
REWARD_ILLEGAL        =  -2.0   

#preprocessing
def preprocess_frame(frame):
    #Convertion de RGB vers niveaux de gris 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

class FrameStack:
    #création d'une mémoire (de 4 frame) pour l'agent pour déduire le mouvement des pions
    def __init__(self, n):
        self.n = n
        self.frames = deque(maxlen=n) #création d'une file pour garder les n dernières frames

    def reset(self, frame):
        processed = preprocess_frame(frame) #on preprocess (BRG2Gris)
        for _ in range(self.n):
            self.frames.append(processed) #on remplie le début de la pile
        return self._get_state()

    def step(self, frame):
        self.frames.append(preprocess_frame(frame)) #implémentation de la file en sortant la dernière frame
        return self._get_state()

    def _get_state(self):
        return np.stack(self.frames, axis=0)  

#réseau de neurones
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        #analyse de l'image en 3 passes "Conv2d" pour récupérer les détails importants
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4), #distinction plateau
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), #distinctions pions
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), #position exact des pions
            nn.ReLU()
        )
        conv_out = self._get_conv_output((FRAME_STACK, *FRAME_SIZE)) #savoir combien de neurones on a besoin pour la couche suivante
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512), #condense les valeurs
            nn.ReLU(),#nettoyage
            nn.Linear(512, n_actions) #réduction à 10 valeurs finales (permet de savoir quelle action est la mieux à faire)
        )

    def _get_conv_output(self, shape): #sert à mesurer la taille de sortie
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x): #permet d'avoir la visualisation, passer au décisionnel et rtourner les scores
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity): #création de la file
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done): #ajoute l'information dans la fille
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): #pioche dans lafile pour entrainer aléatoirement sans repartir de 0 
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return ( #on va transformer en tensor Pytorch avant d'envoyer
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.long),
            torch.tensor(rewards,                dtype=torch.float32),
            torch.tensor(np.array(next_states),  dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

#récompenses personalisées 
def custom_reward(raw_reward, prev_score, new_score, terminated, info):
    #rien ne se passe
    reward = 0.0

    if raw_reward > 0:
        #on a capturé un pion adverse
        reward += REWARD_CAPTURE * raw_reward
    elif raw_reward < 0:
        #action illégale
        reward += REWARD_ILLEGAL
    else:
        #on a pas capturé mais on vérifie si l'adversaire oui
        score_delta = new_score - prev_score
        if score_delta < 0:
            reward += REWARD_CAPTURED * abs(score_delta)

    if terminated:
        score = info.get("score", 0)
        if score > 32:
            reward += REWARD_WIN   # on a plus de la moitié des pions
        elif score < 32:
            reward += REWARD_LOSE  # on a moins de la moitié des pions

    return reward

#définition d l'agent avec l'usage du process en fonction du GPU
class DQNAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions 
        self.epsilon   = EPSILON_START
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")    #apple Mx
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")   #GPU NVIDIA
        else:
            self.device = torch.device("cpu")    #CPU
        print(f" Utilisation du device : {self.device}")

        self.policy_net = DQN(n_actions).to(self.device) #création du réseau principal
        self.target_net = DQN(n_actions).to(self.device) #création d'un secon réseau de référence
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() #on fait en sorte que celui ci n'apprenne pas mais soit juste une référence

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR) #création de l'optimiseur 
        self.memory    = ReplayBuffer(MEMORY_SIZE) #on crée la mémoire
        self.steps     = 0 #on compte les pas

    def select_action(self, state):
        if random.random() < self.epsilon: #choix de si on explore ou non en fonction de epsilon
            return random.randrange(self.n_actions) #choix de l'action de manière aléatoire
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) 
            return self.policy_net(state_t).argmax().item() #choix de l'action en fonction de l'index le plus élevée

    def store(self, *args):
        self.memory.push(*args) 

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None #pas assez de mémoire donc on ne fait rien

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) #on ne récupère que les actions réellement faites
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0] # on estime la meilleure possibilité en fonction de l'état actuelle pour l'état suivant
            target = rewards + GAMMA * next_q * (1 - dones) #on calcule la récompense

        loss = nn.SmoothL1Loss()(q_values, target) #calcul de l'erreur
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# entrainement
def train():
    #on crée l'env et on récupèr les actions, puis on initialise le DQN et on crée la pile de frames
    env = gym.make("ALE/Othello-v5", render_mode=None)
    n_actions = env.action_space.n
    agent     = DQNAgent(n_actions)
    stacker   = FrameStack(FRAME_STACK)

    print(f"  Entraînement DQN sur Othello Atari")
    print(f"  Actions disponibles : {n_actions}")
    print(f"  Épisodes            : {EPISODES}")

    episode_rewards = []

    for episode in range(1, EPISODES + 1):
        #on réinitialise le jeu et on récupère le début
        obs, info   = env.reset()
        #on envoie plusieurs fire pour etre sur du démarage de la partie
        for _ in range(60):
            obs, _, _, _, info = env.step(1)  # FIRE
        for _ in range(20):
            obs, _, _, _, info = env.step(0)  # NOOP
        #on se prépare en initialisant la pile pour la première observation
        state       = stacker.reset(obs)
        total_reward = 0.0
        prev_score  = 0

        for step in range(MAX_STEPS):
            action     = agent.select_action(state) #choix de l'action
            obs, raw_reward, terminated, truncated, info = env.step(action) #on fait l'action, on récupère toutes les infos etc
            next_state = stacker.step(obs) #on ajoute la frame à la pil

            #mise à jour du score et vérification de l'état de la partie
            new_score  = info.get("score", 0)
            reward     = custom_reward(raw_reward, prev_score, new_score, terminated, info)
            prev_score = new_score
            done       = terminated or truncated

            agent.store(state, action, reward, next_state, float(done)) #sauvegarde dans la mémoire
            loss = agent.learn()
            state        = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward) #sauvegarde de la récompense de l'épisode
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY) #diminution de l'epsilon

        if episode % TARGET_UPDATE == 0:
            agent.update_target()
        #impression pour chaque épisode de c qu'il c'est passé
        avg = np.mean(episode_rewards[-10:])
        print(f"Épisode {episode:4d}/{EPISODES} | "
              f"Récompense : {total_reward:8.1f} | "
              f"Moy(10)   : {avg:8.1f} | "
              f"Epsilon   : {agent.epsilon:.3f}")

    env.close()
    print("\nEntraînement terminé !")

    # Sauvegarde du modèle
    torch.save(agent.policy_net.state_dict(), "dqn_othello.pth")
    print(" Modèle sauvegardé : dqn_othello.pth")

    return agent

#demo finale plus enregistrement
def demo(agent, video_path="othello_demo.mp4"):
    print(f"\nLancement de la démo finale")
    env     = gym.make("ALE/Othello-v5", render_mode="rgb_array") #création env
    stacker = FrameStack(FRAME_STACK)

    obs, info = env.reset() #on remet le jeu à 0
    state     = stacker.reset(obs)

    #comme précédemment on envoie plusieurs fire pour vérifier que le jeu se lance puis plusieurs NOOP pour que le jeu s'initialise correctement
    for _ in range(60):
        obs, _, _, _, _ = env.step(1)  #Fire
        state = stacker.step(obs)
    for _ in range(20):
        obs, _, _, _, _ = env.step(0)  #NOOP
        state = stacker.step(obs)

    frames = [] #permet de récupérer la vidéo
    agent.epsilon = 0.0  #on ne veut pas d'exploration dans le dernier donc eps=0 

    for step in range(MAX_STEPS):
        #là on fait la vidéo
        frame = env.render() 
        frames.append(frame)
        #l'agnt joue avec ce qu'il a appris
        action = agent.select_action(state)
        obs, _, terminated, truncated, _ = env.step(action)
        state = stacker.step(obs)

        if terminated or truncated:
            break

    env.close()

    #On récupère la vidéo et on la reconstruit
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    print(f" Vidéo sauvegardée : {video_path} ({len(frames)} frames)")

    #On affiche avec pygame
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((w, h)) #création de la fenêtre
    pygame.display.set_caption("DQN Othello - Démonstration")
    clock  = pygame.time.Clock()

    running = True
    idx     = 0
    while running and idx < len(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        surf = pygame.surfarray.make_surface(frames[idx].swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
        idx += 1

    pygame.quit()

#main
if __name__ == "__main__":
    trained_agent = train()
    demo(trained_agent)