import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from WarehouseEnv_V0 import WarehouseEnv

# Simple agent for demonstration
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


# --- Main Interaction ---
env = WarehouseEnv()

agents = [RandomAgent(env.action_space), RandomAgent(env.action_space)]

num_episodes = 5
for episode in range(num_episodes):
    obs = env.reset()

    done = False
    while not done:
        for agent in agents:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            print(f"Agent action: {action}, Reward: {reward}, Task List: {obs}") 