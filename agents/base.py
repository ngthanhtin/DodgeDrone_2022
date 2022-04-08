from abc import ABC
import numpy as np
import gym


class BaseAgent(ABC):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (2,))

    def select_action(self, obs) -> np.array:
        """
        # Outputs action given the current observation
        obs: a dictionary
            During local development, the participants may specify their desired observations.
            During evaluation on AICrowd, the participants will have access to
            obs =
            {
              'CameraFrontRGB': front_img, # numpy array of shape (width, height, 3)
              'CameraLeftRGB': left_img, # numpy array of shape (width, height, 3)
              'CameraRightRGB': right_img, # numpy array of shape (width, height, 3)
              'track_id': track_id, # integer value associated with a specific racetrack
              'speed': speed # float value of vehicle speed in m/s
            }
        returns:
            action: np.array (2,)
            action should be in the form of [\delta, a], where \delta is the normalized steering angle, and a is the normalized acceleration.
        """
        pass

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        Defaults to select_action
        """
        return self.select_action(obs)

    def training(self, env):
        """
        Training loop
        - Local development OR Stage 2 'practice' phase
        """
        pass

    def load_model(self, path):
        """
        Load model checkpoints.
        """
        pass

    def save_model(self, path):
        """
        Save model checkpoints.
        """
        pass
