import gymnasium as gym

env = gym.make("ALE/Breakout-v5", render_mode="human")

obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()