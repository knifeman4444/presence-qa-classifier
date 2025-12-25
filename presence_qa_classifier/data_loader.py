"""
Data loading and preprocessing class
"""

import logging
from datasets import Dataset
from presence_qa_classifier.config import Config

logger = logging.getLogger(__name__)

# Import med_simple converter
try:
    from data.train.dialogue.med_simple.convert_to_unsloth_format import (
        convert_to_unsloth_format as med_simple_convert_to_unsloth_format,
    )
except ImportError:
    med_simple_convert_to_unsloth_format = None

# Import gorstroy converter (optional)
try:
    from data.train.dialogue.gorstroy_add_info.convert_to_unsloth_format import (
        convert_to_unsloth_format as gorstroy_convert_to_unsloth_format,
    )
except ImportError:
    gorstroy_convert_to_unsloth_format = None


class DataLoader:
    """Data loading and preprocessing class"""

    def __init__(self, config: Config):
        self.config = config

    def load_dataset(self, use_validation: bool = True):
        """Load the raw dataset using conversion functions"""
        logger.info("Loading dataset...")

        try:
            # Choose conversion function
            if getattr(self.config, "data_preparer", "medsimple") == "medsimple":
                if med_simple_convert_to_unsloth_format is None:
                    raise ImportError("med_simple_convert_to_unsloth_format is not available.")
                convert_fn = med_simple_convert_to_unsloth_format
           else:
                raise ValueError(f"Unknown data_preparer: {getattr(self.config, 'data_preparer')}")

            result = convert_fn(include_validation=use_validation)

            train_data, val_data = None, None

            if use_validation:
                if isinstance(result, tuple):
                    train_data, val_data = result
                else:
                    train_data, val_data = result, None
                    logger.warning("Validation data unavailable despite request.")
            else:
                train_data = result

            # Ensure Dataset type
            train_data = self._ensure_dataset(train_data, "train")
            if val_data:
                val_data = self._ensure_dataset(val_data, "validation")

            logger.info(f"Loaded: Train={len(train_data)} samples" + (f", Val={len(val_data)} samples" if val_data else ""))
            return train_data, val_data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise RuntimeError(f"Data loading failed: {e}")

    def _ensure_dataset(self, data, name):
        if data and not isinstance(data, Dataset):
            if isinstance(data, list):
                return Dataset.from_list(data)
            raise ValueError(f"Unexpected type for {name} data")
        return data

    @staticmethod
    def apply_chat_template(dataset: Dataset, tokenizer) -> Dataset:
        """Apply chat template to dataset for Unsloth"""
        logger.info("Applying chat template to dataset...")
        
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
        logger.debug(f"Sample: {formatted_dataset[0]['text'][:100]}...")
        
        return formatted_dataset
