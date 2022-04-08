import json
import time
import numpy as np
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def select_action(self, obs) -> np.array:
        return self.action_space.sample()

    def register_reset(self, obs) -> np.array:
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def training(self, env):
        """Train your agent here."""
        for _ in range(300):
            done = False
            obs, _ = env.reset()

            while not done:
                action = self.select_action(obs)
                obs, reward, done, info = env.step(action)

            # Update your agent
