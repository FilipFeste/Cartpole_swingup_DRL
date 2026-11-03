#Training an agent with SB3 and stores the model
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import cartpole_swingup

ENV_ID = "CartPoleSwingUp-v0"
SEED = 0

def main():
    torch.set_num_threads(1)  # currently running other simulations..
    #Flere miljøer parallelt gir mer variert data per oppdatering
    env = make_vec_env(ENV_ID, n_envs=8, seed=SEED)
    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED + 1000)

    #Evaluerer policyen jevnlig og lagrer den beste modellen
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env, 
        seed= SEED,
        verbose=1,
        tensorboard_log="logs/",
        device="cpu",
    )

    model.learn(total_timesteps=1_000_000, callback=eval_callback)
    model.save("models/ppo_final")


if __name__ == "__main__":
    main()


