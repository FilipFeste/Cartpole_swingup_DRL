#Cartpole environment for swingup task
#Self designed

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class CartPoleSwingUpEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        #Vil at pendelen skal kunne svinge fritt, bare banegrensen avslutter episoden
        self.theta_threshold_radians = math.inf

        #Observasjon: [x, x_dot, cos(theta), sin(theta), theta_dot]
        high = np.array([
            self.x_threshold * 2, 
            np.inf,
            1.0,
            1.0,
            np.inf],
            dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)  # bare seeding av self.np_random
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[2] += math.pi        # start hengende
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        _, _, terminated, truncated, info = super().step(action)
        x, x_dot, theta, theta_dot = self.state

        reward = (1 + math.cos(theta)) / 2  

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)


gym.register(
    id="CartPoleSwingUp-v0",
    entry_point=CartPoleSwingUpEnv,
    max_episode_steps=500,
)