"""
Model training class and main execution function
"""

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import logging
import mlflow
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
from pathlib import Path
from typing import Optional

from presence_qa_classifier.data_loader import DataLoader

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training class"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Setup model and tokenizer"""
        model_name = self.config.models.model_name
        logger.info(f"Loading model: {model_name}")

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.config.models.max_seq_length,
            load_in_4bit=self.config.models.load_in_4bit,
            load_in_8bit=self.config.models.load_in_8bit,
            full_finetuning=self.config.models.full_finetuning,
        )

        self.model = FastModel.get_peft_model(
            self.model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.config.models.lora.r,
            lora_alpha=self.config.models.lora.alpha,
            # lora_dropout=self.config.models.lora.dropout,
            # bias=self.config.models.lora.bias,
            random_state=self.config.random_state,
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=self.config.models.model_short_name,
        )

        logger.info("Model and tokenizer loaded.")

    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Setup SFT trainer"""
        logger.info("Setting up SFT trainer...")

        output_dir = Path(self.config.models.output_dir) / self.config.project_name
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=self.config.models.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.models.validation.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.models.training.gradient_accumulation_steps,
            warmup_steps=self.config.models.training.warmup_steps,
            max_steps=self.config.models.training.max_steps,
            learning_rate=self.config.models.training.learning_rate,
            logging_steps=self.config.models.training.logging_steps,
            optim=self.config.models.training.optim,
            lr_scheduler_type=self.config.models.training.lr_scheduler_type,
            seed=self.config.random_state,
            output_dir=str(output_dir),
            report_to="mlflow" if not self.config.logging.no_mlflow else "none",
            dataset_num_proc=self.config.training_process.training.dataset_num_proc,
            eval_strategy=self.config.training_process.validation.eval_strategy,
            eval_steps=self.config.models.validation.eval_steps,
            save_strategy=self.config.training_process.validation.save_strategy,
            save_steps=self.config.models.validation.save_steps,
            save_total_limit=self.config.training_process.validation.save_total_limit,
            load_best_model_at_end=self.config.training_process.validation.load_best_model_at_end,
            metric_for_best_model=self.config.training_process.validation.metric_for_best_model,
            greater_is_better=self.config.training_process.validation.greater_is_better,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.training_process.training.early_stopping_patience)]
        )

        # Train on responses only (Gemma specific)
        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        logger.info("Trainer setup complete.")

    def train(self):
        """Train the model"""
        logger.info("Starting training...")
        
        resume_from_checkpoint = self.config.training_process.training.resume_from_checkpoint  
        trainer_stats = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        logger.info(f"Training completed. Runtime: {trainer_stats.metrics['train_runtime']:.2f}s")
        
        return trainer_stats

    def test(self, test_dataset: Dataset):
        """Evaluate the model on the test dataset"""
        logger.info("Starting evaluation on test dataset...")
        
        if test_dataset is None or len(test_dataset) == 0:
            logger.warning("Test dataset is empty. Skipping evaluation.")
            return {}

        # The trainer.evaluate method expects an eval_dataset argument.
        # We pass the test_dataset as the eval_dataset to reuse the evaluation logic.
        metrics = self.trainer.evaluate(eval_dataset=test_dataset)
        
        # Rename 'eval_' keys to 'test_' for clarity in logging
        test_metrics = {k.replace("eval_", "test_"): v for k, v in metrics.items()}
        
        logger.info(f"Test metrics: {test_metrics}")
        
        if not self.config.logging.no_mlflow:
             mlflow.log_metrics(test_metrics)
             
        return test_metrics

    def save_model(self, output_dir: Optional[Path] = None):
        """Save the trained model"""
        logger.info("Saving model...")
        if output_dir is None:
            output_dir = Path(self.config.models.output_dir) / self.config.project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained_merged(output_dir, self.tokenizer)


def train(cfg: DictConfig):
    """Main training function called by Hydra/Fire"""

    # Initialize MLflow
    if not cfg.logging.no_mlflow:
        mlflow.set_tracking_uri(cfg.logging.mlflow_url)
        mlflow.set_experiment(cfg.project_name)
        mlflow.start_run(run_name=cfg.experiment_description)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    try:
        # 1. Initialize Model & Tokenizer
        trainer = ModelTrainer(cfg)
        trainer.setup_model()

        # 2. Load Raw Data
        data_loader = DataLoader(cfg)
        # Unpack datasets (Train, Val, Test)
        train_dataset, eval_dataset, test_dataset = data_loader.load_datasets()
        # 3. Apply Chat Template (requires tokenizer from trainer)
        train_dataset, eval_dataset, test_dataset = data_loader.apply_chat_template(
            [train_dataset, eval_dataset, test_dataset],
            trainer.tokenizer
        )

        # 4. Setup Trainer with formatted data
        trainer.setup_trainer(train_dataset, eval_dataset)

        # 5. Train
        trainer.train()

        # 6. Test
        trainer.test(test_dataset)

        # 7. Save
        trainer.save_model()

        logger.info("Training process finished successfully.")

    except Exception as e:
        logger.error(f"Critical error during training: {e}", exc_info=True)
        raise
    finally:
        if not cfg.logging.no_mlflow:
            mlflow.end_run()
