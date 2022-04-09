import timeout_decorator
from loguru import logger

from evaluator.evaluator import Evaluator
from config import agent_config, env_sim_config

import argparse

def run_evaluation(args):
    if args.train:
        num_envs = args.n_envs
    else:
        num_envs = 1
    evaluator = Evaluator(
        num_envs=num_envs,
        agent_config=agent_config,
        env_sim_config=env_sim_config,
    )

    # Init evaluator
    evaluator.create_env()
    evaluator.init_agent()

    if args.train: # Training
        try:
            evaluator.train()
        except timeout_decorator.TimeoutError:
            logger.info("Stopping pre-evaluation run")
    else: # Evaluate
        scores = evaluator.evaluate()
        logger.success(f"Average metrics: {scores}")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, help="1: Train, 0: Evaluate")
    parser.add_argument("--n_envs", type=int, default=1, help="100: Train, 1: Evaluate")
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    run_evaluation(args)
