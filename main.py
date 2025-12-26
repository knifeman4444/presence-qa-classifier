import fire
from hydra import compose, initialize
from omegaconf import OmegaConf
from presence_qa_classifier.model_trainer import train
import logging


def configure_logging(log_level="INFO"):
    """
    Configure logging for the entire application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=numeric_level,
        force=True,
    )


def run_train(
    config_name="conf", 
    config_path="configs", 
    overrides=None,
    log_level="INFO"
):
    """
    Run training using Hydra configuration.
    
    Args:
        config_name: Name of the config file (without .yaml)
        config_path: Path to the config directory relative to the script
        overrides: List of overrides (e.g. ["training.max_steps=100", "logging.no_wandb=true"])
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    configure_logging(log_level)
    
    if overrides is None:
        overrides = []
        
    # Initialize Hydra
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
        logger.info(OmegaConf.to_yaml(cfg))
        train(cfg)


def main():
    fire.Fire({
        "train": run_train
    })


if __name__ == "__main__":
    main()
