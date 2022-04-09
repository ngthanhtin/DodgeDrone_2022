from typing import Type
from importlib_resources import path
import numpy as np
from loguru import logger
import timeout_decorator

import os
from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump

import cv2
import torch

def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Evaluator:
    """Evaluator class which consists of a 1-hour pre-evaluation phase followed by an evaluation phase."""

    def __init__(
        self, num_envs, agent_config, env_sim_config):
        logger.info("Starting learn to flghtmare framework")
        env_sim_config["simulation"]["num_envs"] = num_envs
        self.agent_config = agent_config
        self.env_sim_config = env_sim_config

        self.agent = None
        self.env = None

    def init_agent(self):
        self.agent = self.agent_config.agent(self.num_envs)

    def load_agent_model(self, path):
        self.agent.load_model(path)

    def save_agent_model(self, path):
        self.agent.save_model(path)

    def train(self, render=False):
        logger.info("Starting 'practice' phase")
        # connect unity
        if render:
            print("Connect Unity")
            self.env.connectUnity()
        # train   
        path_name = "/home/tinvn/TIN/Drone_Challenge/src/agile_flight/envtest/python2/results/SAC_episode_770.statedict"
        self.agent.load_model(path=path_name)
        self.agent.training(self.env)
        if render:
            print("Disconnecting Unity")
            self.env.disconnectUnity()

    def evaluate(self):
        """Evaluate the episodes."""

        path_name = "/home/tinvn/TIN/Drone_Challenge/src/agile_flight/envtest/python2/results/SAC_episode_770.statedict"
        self.agent.load_model(path=path_name)
        
        logger.info("Starting evaluation")
        episode_count = 0

        obs_dim = self.env.obs_dim
        act_dim = self.env.act_dim
        num_env = self.env.num_envs
        
        # connect unity
        print("Connect Unity")
        self.env.connectUnity()

        # while self.laps_completed < self.env_config.n_eval_laps:
        for i in range(10):
            state = self.env.reset()
            self.agent.register_reset(state)
            
            dummy_actions = np.random.rand(self.num_envs, act_dim) * 2 - np.ones(shape=(self.num_envs, act_dim))
            camera, features, state2, r, d, info = self.agent._step(self.env, dummy_actions)

            while len(np.nonzero(d)[0]) == 0:
                action = self.agent.select_action(features, encode=False)
                camera, features, state, r, d, info = self.agent._step(self.env, action)
                
                
            episode_count += 1
            logger.info(
                f"Completed episode: {episode_count} : {info}"
            )

        print("Disconnecting Unity")
        self.env.disconnectUnity()

        return info

    def create_env(self):
        # to connect unity
        self.env_sim_config["unity"]["render"] = "yes"
        # to simulate rgb camera
        self.env_sim_config["rgb_camera"]["on"] = "yes"
        # create training environment
        self.num_envs = self.env_sim_config["simulation"]["num_envs"]
        # load the Unity standardalone, make sure you have downloaded it.
        os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")
        self.env = VisionEnv_v1(dump(self.env_sim_config, Dumper=RoundTripDumper), False)
        self.env = wrapper.FlightEnvVec(self.env)
        self.env.reset(random=True)

        # set random seed
        configure_random_seed(1024, env=self.env)
        
