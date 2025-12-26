"""
Data loading and preprocessing class
"""

import logging
from typing import Tuple, Optional, List
from datasets import Dataset
from omegaconf import DictConfig
import random

from presence_qa_classifier.data_preparation.boolq_preparation import load_boolq_dataset
from presence_qa_classifier.data_preparation.mnli_preparation import load_mnli_dataset

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and preprocessing class"""

    def __init__(self, config: DictConfig):
        self.config = config

    def load_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Load datasets from config.
        For each dataset in config.datasets:
          - Load train and test (if available)
          - Merge train and test files separately
          - Split train into actual train and validation
          - Return (actual_train_dataset, validation_dataset, test_dataset)
        """
        logger.info("Loading datasets...")

        all_train_conversations = []
        all_test_conversations = []

        # Iterate over datasets in the config list
        # config.datasets is a list of dataset configs
        
        for dataset_cfg in self.config.datasets.values():
            logger.info(f"Processing dataset: {dataset_cfg.name}")
            
            raw_data = None
            if dataset_cfg.data_preparer == "boolq":
                 raw_data = load_boolq_dataset(
                     dataset_cfg.data_path,
                     dataset_cfg.system_prompt_path,
                     dataset_cfg.test_data_path
                 )
            elif dataset_cfg.data_preparer == "mnli":
                 raw_data = load_mnli_dataset(
                     dataset_cfg.data_path,
                     dataset_cfg.system_prompt_path,
                     dataset_cfg.test_data_path
                 )
            else:
                 raise ValueError(f"Unknown data_preparer: {dataset_cfg.data_preparer}")
            # raw_data is either List (Train Only) or Tuple(Train List, Test List)
            
            ds_train = []
            ds_test = []

            if isinstance(raw_data, tuple):
                ds_train = raw_data[0]
                ds_test = raw_data[1]
            elif isinstance(raw_data, list):
                ds_train = raw_data
                ds_test = [] # No test data provided for this source
            else:
                 raise ValueError(f"Unexpected data format from loader for {dataset_cfg.name}")

            # Take subset of training data if specified
            if dataset_cfg.get("training_subset_size", None):
                ds_train = random.sample(ds_train, dataset_cfg.training_subset_size)

            all_train_conversations.extend(ds_train)
            all_test_conversations.extend(ds_test)
            
            logger.info(f"  Added {len(ds_train)} train samples, {len(ds_test)} test samples.")

        # Shuffle all training conversations
        random.shuffle(all_train_conversations)
        # Create HF Datasets
        combined_train_ds = Dataset.from_dict({"conversations": all_train_conversations})
        
        # If we have test data
        combined_test_ds = None
        if all_test_conversations:
            combined_test_ds = Dataset.from_dict({"conversations": all_test_conversations})

        logger.info(f"Total Combined Train: {len(combined_train_ds)}")
        logger.info(f"Total Combined Test: {len(combined_test_ds)}")

        val_size = self.config.training_process.validation.val_size
        split = combined_train_ds.train_test_split(test_size=val_size, shuffle=True, seed=self.config.random_state)
        actual_train_ds = split["train"]
        validation_ds = split["test"]
        
        logger.info(f"Final sizes: Train={len(actual_train_ds)}, Val={len(validation_ds)}, Test={len(combined_test_ds)}")    

        return actual_train_ds, validation_ds, combined_test_ds

    @staticmethod
    def apply_chat_template(datasets: List[Dataset], tokenizer) -> List[Dataset]:
        """Apply chat template to dataset for Unsloth"""
        logger.info("Applying chat template to datasets...")
        
        formatted_datasets = []
        for dataset in datasets:
            # Check first item to ensure structure
            if not dataset or 'conversations' not in dataset.column_names:
                logger.warning("Dataset empty or missing 'conversations' column.")
                return dataset

            texts = [
                {
                    "text": tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    ).removeprefix("<bos>")
                }
                for convo in dataset["conversations"]
            ]

            formatted_dataset = Dataset.from_list(texts)
            logger.info(f"Dataset formatted. Size: {len(formatted_dataset)}")
            if len(formatted_dataset) > 0:
                logger.debug(f"Sample: {formatted_dataset[0]['text'][:100]}...")
            formatted_datasets.append(formatted_dataset)
        
        return formatted_datasets
