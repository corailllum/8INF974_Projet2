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
import pygame

#définition des paramètres
EPISODES      = 1000
MAX_STEPS     = 10000
BATCH_SIZE    = 32
GAMMA         = 0.99
LR            = 1e-4
MEMORY_SIZE   = 50_000
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.995      #décroissance par épisode
TARGET_UPDATE = 10
FRAME_STACK   = 4
FRAME_SIZE    = (84, 84)

#récompenses : atari pong retourne +1 si on marque, -1 si on encaisse
REWARD_POINT_WIN  =  1.0
REWARD_POINT_LOSE = -1.0

#preprocessing : conversion rgb vers niveaux de gris
def preprocess_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

class FrameStack:
    #pile de 4 frames pour donner un contexte temporel à l'agent
    def __init__(self, n):
        self.n      = n
        self.frames = deque(maxlen=n)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.n):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame):
        self.frames.append(preprocess_frame(frame))
        return self._get_state()

    def _get_state(self):
        return np.stack(self.frames, axis=0)  #(4, 84, 84)

#réseau de neurones : architecture classique pour atari
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        #3 couches convolutionnelles pour analyser les frames
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),  #extraction des formes grossières
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            #détails intermédiaires
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            #détails fins
            nn.ReLU()
        )
        conv_out = self._get_conv_output((FRAME_STACK, *FRAME_SIZE))
        #2 couches linéaires pour la décision
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#replay buffer : mémoire de l'agent
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

#agent dqn
class DQNAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.epsilon   = EPSILON_START

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")    #apple silicon
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")   #gpu nvidia
        else:
            self.device = torch.device("cpu")    #cpu fallback
        print(f" Device : {self.device}")

        self.policy_net = DQN(n_actions).to(self.device)  #réseau principal
        self.target_net = DQN(n_actions).to(self.device)  #réseau de référence
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory    = ReplayBuffer(MEMORY_SIZE)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)  #exploration aléatoire
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy_net(state_t).argmax().item()  #exploitation

    def store(self, *args):
        self.memory.push(*args)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None  #pas assez de souvenirs

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

#boucle d'entraînement
def train():
    #on utilise directement l'env atari sans aucun plateau miroir
    env       = gym.make("ALE/Pong-v5", render_mode=None)
    n_actions = env.action_space.n
    agent     = DQNAgent(n_actions)
    stacker   = FrameStack(FRAME_STACK)

    print(f"  Entraînement DQN sur Pong Atari")
    print(f"  Actions disponibles : {n_actions}")
    print(f"  Épisodes : {EPISODES}")

    episode_rewards = []

    for episode in range(1, EPISODES + 1):
        obs, info    = env.reset()
        state        = stacker.reset(obs)
        total_reward = 0.0
        points_gagnes = 0
        points_perdus = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = stacker.step(obs)
            done = terminated or truncated

            #on compte les points pour l'affichage
            if reward > 0:
                points_gagnes += 1
            elif reward < 0:
                points_perdus += 1

            agent.store(state, action, reward, next_state, float(done))
            agent.learn()

            state        = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        #résultat de la partie (pong se joue en 21 points)
        if total_reward > 0:
            resultat = "GAGNE "
        elif total_reward < 0:
            resultat = "PERDU "
        else:
            resultat = "NUL   "

        avg = np.mean(episode_rewards[-10:])
        print(f"Épisode {episode:4d}/{EPISODES} | "
              f"Récompense : {total_reward:7.1f} | "
              f"Moy(10) : {avg:7.1f} | "
              f"Epsilon : {agent.epsilon:.3f} | "
              f"{resultat} ({points_gagnes}-{points_perdus})")

    env.close()
    print("\nEntraînement terminé !")
    torch.save(agent.policy_net.state_dict(), "dqn_pong.pth2")
    print(" Modèle sauvegardé : dqn_pong.pth2")

    return agent

#démo finale avec affichage pygame et enregistrement vidéo
def demo(agent, video_path="pong_demo2.mp4"):
    print("\nLancement de la démo finale")

    env     = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    stacker = FrameStack(FRAME_STACK)

    obs, info = env.reset()
    state     = stacker.reset(obs)
    agent.epsilon = 0.0  #pas d'exploration en démo

    frames = []

    pygame.init()
    frame0 = env.render()
    h, w   = frame0.shape[:2]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("DQN Pong - Démonstration")
    clock  = pygame.time.Clock()

    running      = True
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #capture et affichage de la frame atari
        frame = env.render()
        frames.append(frame)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)  #30 fps

        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        state         = stacker.step(obs)
        total_reward += reward

        if terminated or truncated:
            running = False

    print(f"\n Résultat final : {total_reward}")
    pygame.quit()
    env.close()

    #sauvegarde vidéo
    if frames:
        h, w   = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f" Vidéo sauvegardée : {video_path} ({len(frames)} frames)")

#main
if __name__ == "__main__":
    trained_agent = train()
    demo(trained_agent)
