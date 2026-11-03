import gymnasium as gym
import numpy as np
from pathlib import Path
from gymnasium.utils.env_checker import check_env
import cartpole_swingup

check_env(gym.make("CartPoleSwingUp-v0").unwrapped)  # Check if the environment follows the Gymnasium API
env = gym.make("CartPoleSwingUp-v0", render_mode="None")

n_episodes = 20
returns = []

env.action_space.seed(0)
for ep in range(n_episodes):
    obs, info = env.reset(seed = ep)
    episode_return = 0.0
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        done = terminated or truncated

    returns.append(episode_return)

env.close()

returns = np.array(returns)
print(f"Random baseline: {returns.mean():.1f} ± {returns.std():.1f} over {n_episodes} episoder")

Path("results").mkdir(exist_ok=True)                    # │
np.savetxt("results/random_baseline.csv", returns, delimiter=",",
           header="episode_return", comments="")  


