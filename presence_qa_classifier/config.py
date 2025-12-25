"""
Configuration class for training parameters
"""

import logging
from typing import Any, Dict, List, Optional


class Config:
    """Configuration class for training parameters"""

    def __init__(self, project_name: str, experiment_description: str):
        # Model configuration - Only Gemma
        self.model_name = "unsloth/gemma-3-12b-it"
        self.model_short_name = "gemma-3"
        
        self.max_seq_length = 8192
        self.load_in_4bit = True
        self.load_in_8bit = False
        self.full_finetuning = False

        # LoRA configuration
        self.lora_r = 4
        self.lora_alpha = self.lora_r
        self.lora_dropout = 0
        self.lora_bias = "none"
        self.random_state = 3407

        # Training configuration
        self.per_device_train_batch_size = 2
        self.gradient_accumulation_steps = 64
        self.warmup_steps = 8
        self.max_steps = 40
        self.learning_rate = 1e-3
        self.weight_decay = 0.01
        self.lr_scheduler_type = "linear"
        self.optim = "adamw_8bit"
        self.logging_steps = 1
        self.dataset_num_proc = 2

        # Validation configuration
        self.use_validation = True
        self.eval_steps = 3
        self.save_steps = 10
        self.save_total_limit = 2
        self.per_device_eval_batch_size = 2
        self.eval_strategy = "steps"
        self.save_strategy = "steps"
        self.load_best_model_at_end = False
        self.metric_for_best_model = "eval_loss"
        self.greater_is_better = False

        # Data paths
        self.data_path = "data/train/dialogue/med_simple"
        self.models_output_dir = "models/raw"
        self.no_wandb = False

        # Data loading selector
        self.data_preparer = "medsimple"

        # Experiment configuration
        self.project_name = project_name
        self.experiment_name = "med_simple-20250826_170003"  # Consider making this dynamic in main
        self.experiment_description = experiment_description

        # Inference configuration (kept minimal for potential future use, though test_inference is removed)
        self.temperature = 1.0
        self.top_p = 0.95
        self.top_k = 64
        self.max_new_tokens = 8192

        self.resume_from_checkpoint = True

    @classmethod
    def from_yaml(cls, y: Dict[str, Any]) -> "Config":
        """Create Config from YAML dict"""
        project_name = y.get("project_name")
        experiment_description = y.get("experiment_description")
        if not project_name or not experiment_description:
            raise ValueError(
                "YAML must include 'project_name' and 'experiment_description'"
            )

        cfg = cls(project_name, experiment_description)

        # Allowed overrides
        overridable = [
            "model_name", "max_seq_length", "load_in_4bit", "load_in_8bit", "full_finetuning",
            "lora_r", "lora_alpha", "lora_dropout", "lora_bias", "random_state",
            "per_device_train_batch_size", "gradient_accumulation_steps", "warmup_steps",
            "max_steps", "learning_rate", "weight_decay", "lr_scheduler_type", "optim",
            "logging_steps", "dataset_num_proc",
            "use_validation", "eval_steps", "save_steps", "save_total_limit",
            "per_device_eval_batch_size", "eval_strategy", "save_strategy",
            "load_best_model_at_end", "metric_for_best_model", "greater_is_better",
            "data_path", "models_output_dir", "no_wandb", "experiment_name",
            "temperature", "top_p", "top_k", "max_new_tokens",
            "data_preparer", "resume_from_checkpoint",
        ]

        for k in overridable:
            if k in y:
                setattr(cfg, k, y[k])

        return cfg
