"""
Main training script for Unsloth LoRA fine-tuning
"""

import argparse
import os
import logging
import sys

import dotenv
import wandb

from presence_qa_classifier.config import Config
from presence_qa_classifier.data_loader import DataLoader
from presence_qa_classifier.model_trainer import ModelTrainer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Unsloth LoRA fine-tuning")
    parser.add_argument("--project-name", type=str, help="Name for the project", required=True)
    parser.add_argument("--experiment-description", type=str, help="Description for the experiment", required=True)
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Per device batch size")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no-validation", action="store_true", help="Disable validation during training")

    args = parser.parse_args()

    logger.info("Initializing configuration...")
    config = Config(args.project_name, args.experiment_description)

    # Overrides
    if args.max_steps: config.max_steps = args.max_steps
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.batch_size: config.per_device_train_batch_size = args.batch_size
    if args.no_validation: config.use_validation = False
    if args.no_wandb: config.no_wandb = True

    # Initialize wandb
    if not args.no_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            entity=os.getenv("WANDB_TEAM_NAME"),
            project=args.project_name,
            name=config.experiment_name,
            config=vars(config) # Dump simple attributes
        )

    try:
        # 1. Initialize Model & Tokenizer
        trainer = ModelTrainer(config)
        trainer.setup_model()

        # 2. Load Raw Data
        data_loader = DataLoader(config)
        train_dataset, eval_dataset = data_loader.load_dataset(use_validation=config.use_validation)
        
        # 3. Apply Chat Template (requires tokenizer from trainer)
        formatted_train_dataset = DataLoader.apply_chat_template(train_dataset, trainer.tokenizer)
        formatted_eval_dataset = DataLoader.apply_chat_template(eval_dataset, trainer.tokenizer) if eval_dataset else None

        # 4. Setup Trainer with formatted data
        trainer.setup_trainer(formatted_train_dataset, formatted_eval_dataset)

        # 5. Train
        trainer.train()

        # 6. Save
        trainer.save_model()

        logger.info("Training process finished successfully.")

    except Exception as e:
        logger.error(f"Critical error during training: {e}", exc_info=True)
        raise
    finally:
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
