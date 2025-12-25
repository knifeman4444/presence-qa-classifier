"""
Model training class
"""

import json
import os
import logging

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from presence_qa_classifier.config import Config

logger = logging.getLogger(__name__)

# Fix PyTorch compilation issues
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.suppress_errors = True


class ModelTrainer:
    """Model training class"""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Setup model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            full_finetuning=self.config.full_finetuning,
        )

        self.model = FastModel.get_peft_model(
            self.model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            random_state=self.config.random_state,
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=self.config.model_short_name,
        )

        logger.info("Model and tokenizer loaded.")

    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Setup SFT trainer"""
        logger.info("Setting up SFT trainer...")

        output_dir = os.path.join(
            self.config.models_output_dir, self.config.experiment_name
        )
        os.makedirs(output_dir, exist_ok=True)

        use_evaluation = eval_dataset is not None and self.config.use_validation

        training_args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size if use_evaluation else None,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.random_state,
            output_dir=str(output_dir),
            report_to="wandb" if not self.config.no_wandb else "none",
            dataset_num_proc=self.config.dataset_num_proc,
            eval_strategy=self.config.eval_strategy if use_evaluation else "no",
            eval_steps=self.config.eval_steps if use_evaluation else None,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if use_evaluation else None,
            args=training_args,
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
        
        self._log_memory_usage("Start")
        
        trainer_stats = self.trainer.train(
            resume_from_checkpoint=self.config.resume_from_checkpoint
        )
        
        self._log_memory_usage("End")
        logger.info(f"Training completed. Runtime: {trainer_stats.metrics['train_runtime']:.2f}s")
        
        return trainer_stats

    def save_model(self, output_dir: str = None, save_gguf: bool = True, save_train_config: bool = True):
        """Save the trained model"""
        logger.info("Saving model...")

        if output_dir is None:
            output_dir = os.path.join(
                self.config.models_output_dir, self.config.experiment_name
            )
        os.makedirs(output_dir, exist_ok=True)

        if save_gguf:
            logger.info("Saving in merged and GGUF format...")
            self.model.save_pretrained_merged(output_dir, self.tokenizer)
            self.model.save_pretrained_gguf(
                output_dir,
                tokenizer=self.tokenizer,
                quantization_type="q8_0",
            )
            logger.info("GGUF format saved.")
        else:
            logger.info("Saving in merged_16bit format...")
            self.model.save_pretrained_merged(
                output_dir, self.tokenizer, save_method="merged_16bit"
            )
            
            hf_username = os.getenv("HF_USERNAME")
            if hf_username:
                repo_name = f"{hf_username}/{self.config.experiment_name}"
                self.model.push_to_hub_merged(
                    repo_name,
                    self.tokenizer,
                    save_method="merged_16bit",
                    token=os.getenv("HF_TOKEN"),
                )
                logger.info(f"Pushed to HF Hub: {repo_name}")
            else:
                logger.warning("HF_USERNAME not set. Skipping push.")

        if save_train_config:
            config_path = os.path.join(output_dir, "training_config.json")
            config_dict = {
                "model_name": self.config.model_name,
                "experiment_name": self.config.experiment_name,
                "lora_r": self.config.lora_r,
                "max_steps": self.config.max_steps,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.per_device_train_batch_size,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Config saved to {config_path}")

    def _log_memory_usage(self, phase: str):
        if torch.cuda.is_available():
            reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            logger.info(f"[{phase}] GPU Memory Reserved: {reserved} GB")
