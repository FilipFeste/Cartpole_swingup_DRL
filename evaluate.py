#Loading the model, running the model and render episodes

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

import cartpole_swingup  # registrerer CartPoleSwingUp-v0

ENV_ID = "CartPoleSwingUp-v0"
MODEL_PATH = "models/best_model"


def evaluate(model, n_episodes=20):
    env = gym.make(ENV_ID)
    returns = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)  # samme seeds som baselinen
        episode_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
    env.close()
    return np.array(returns)


def record_gif(policy, path, seed=0, frame_skip=2):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    step = 0
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        if step % frame_skip == 0:
            frames.append(env.render())
    env.close()

    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=20 * frame_skip,  # ms per bilde: 0.02 s per steg = sanntid
        loop=0,                    # gjenta i det uendelige
    )


def main():
    model = PPO.load(MODEL_PATH)
    Path("results").mkdir(exist_ok=True)

    returns = evaluate(model)
    print(f"PPO: {returns.mean():.1f} ± {returns.std():.1f} over {len(returns)} episoder")
    np.savetxt("results/ppo_eval.csv", returns, delimiter=",",
               header="episode_return", comments="")

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    random_space = gym.spaces.Discrete(2, seed=0)

    def random_policy(obs):
        return int(random_space.sample())  # ignorerer obs, akkurat som i baselinen

    record_gif(ppo_policy, "results/ppo_swingup.gif")
    record_gif(random_policy, "results/random_swingup.gif")


if __name__ == "__main__":
    main()