import timeout_decorator
from loguru import logger

from evaluator.evaluator import Evaluator
from config import agent_config, env_sim_config

def run_evaluation(train=False):
    evaluator = Evaluator(
        agent_config=agent_config,
        env_sim_config=env_sim_config,
    )

    # Init evaluator
    evaluator.create_env()
    evaluator.init_agent()

    if train: # Training
        try:
            evaluator.train()
        except timeout_decorator.TimeoutError:
            logger.info("Stopping pre-evaluation run")
    else: # Evaluate
        scores = evaluator.evaluate()
        logger.success(f"Average metrics: {scores}")

if __name__ == "__main__":
    run_evaluation(train=True)
