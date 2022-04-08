import os
# from agents.random_agent import RandomAgent
from agents.sac_agent import SACAgent
from ruamel.yaml import YAML, RoundTripDumper, dump

num_envs = 100
# load for environment and simulation configurations
env_sim_config = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)
env_sim_config["simulation"]["num_envs"] = num_envs

class agent_config():
    agent = SACAgent

