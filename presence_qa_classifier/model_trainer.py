"""
Model training class and main execution function
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from presence_qa_classifier.data_loader import DataLoader
from presence_qa_classifier.metrics import calculate_accuracy


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

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.config.models.max_seq_length,
            load_in_4bit=self.config.models.load_in_4bit,
            load_in_8bit=self.config.models.load_in_8bit,
            full_finetuning=self.config.models.full_finetuning,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.config.models.lora.r,
            lora_alpha=self.config.models.lora.alpha,
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
        early_stopping_patience = self.config.training_process.training.early_stopping_patience

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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
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
        trainer_stats = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info(f"Training completed. Runtime: {trainer_stats.metrics['train_runtime']:.2f}s")

        return trainer_stats

    def test(self, test_dataset: Dataset) -> Dict[str, float]:
        """
        Ugly test of the model just manually generating the answer.

        Args:
            test_dataset: Dataset containing 'conversations' column

        Returns:
            Dictionary of metrics
        """
        logger.info("Starting generation testing...")

        if test_dataset is None or len(test_dataset) == 0:
            logger.warning("Test dataset is empty. Skipping evaluation.")
            return {}

        FastLanguageModel.for_inference(self.model)

        predictions: List[str] = []
        references: List[str] = []

        for item in tqdm(test_dataset, desc="Generating"):
            conversation = item.get("conversations")
            if not conversation:
                continue

            # Last message is the assistant response (ground truth)
            ground_truth = conversation[-1]["content"]

            # Everything before is the prompt
            prompt_msgs = conversation[:-1]

            # Prepare prompt
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer([prompt_text], return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs, max_new_tokens=4, use_cache=True, pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode only new tokens
            generated_ids = outputs[:, inputs.input_ids.shape[-1] :]
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = generated_text.strip().lower()
            ref = ground_truth.strip().lower()
            predictions.append(pred)
            references.append(ref)

        accuracy = calculate_accuracy(predictions, references)
        accuracy_yes_no = calculate_accuracy(predictions, references, filter_labels=["yes", "no"])
        hallucination_rate = 1 - calculate_accuracy(predictions, references, filter_labels=["idk"])

        metrics = {
            "test_accuracy": accuracy,
            "test_accuracy_yes_no": accuracy_yes_no,
            "test_hallucination_rate": hallucination_rate,
        }

        logger.info(f"Test Generation Metrics: {metrics}")

        if predictions:
            logger.info(f"Example 1 - Pred: '{predictions[0]}', Ref: '{references[0]}'")
            if len(predictions) > 1:
                logger.info(f"Example 2 - Pred: '{predictions[1]}', Ref: '{references[1]}'")

        if not self.config.logging.no_mlflow:
            mlflow.log_metrics(metrics)

        return metrics

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
        raw_train_dataset, raw_eval_dataset, raw_test_dataset = data_loader.load_datasets()

        # 3. Apply Chat Template (requires tokenizer from trainer)
        train_dataset, eval_dataset = data_loader.apply_chat_template(
            [raw_train_dataset, raw_eval_dataset], trainer.tokenizer
        )

        # 4. Setup Trainer with formatted data
        trainer.setup_trainer(train_dataset, eval_dataset)

        # 5. Train
        trainer.train()

        # 6. Test
        trainer.test(raw_test_dataset)

        # 7. Save
        trainer.save_model()

        logger.info("Training process finished successfully.")

    except Exception as e:
        logger.error(f"Critical error during training: {e}", exc_info=True)
        raise
    finally:
        if not cfg.logging.no_mlflow:
            mlflow.end_run()
