import json
from pathlib import Path
from typing import Dict, List, Union


def load_system_prompt(system_prompt_path: Union[str, Path]) -> str:
    """Load the system prompt from the data directory."""
    prompt_path = Path(system_prompt_path)

    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise FileNotFoundError(f"System prompt file not found: {prompt_path}")


def format_conversation(item: Dict, system_prompt: str) -> List[Dict[str, str]]:
    """Format a single item into a conversation."""

    sentence1 = item.get("sentence1", "")
    sentence2 = item.get("sentence2", "")
    gold_label = item.get("gold_label", "")
    if gold_label != "contradiction":
        gold_label = "consistent"

    # Format user input with system prompt
    user_content = f'{system_prompt}\n\nPremise: "{sentence1}"\nHypothesis: "{sentence2}"'

    return [{"role": "user", "content": user_content}, {"role": "assistant", "content": gold_label}]


def process_jsonl(file_path: Path, system_prompt: str) -> List[List[Dict[str, str]]]:
    """Read JSONL file and convert to list of conversations."""
    conversations = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # MNLI sometimes has '-' as gold_label for unlabeled/consensus issues,
            # usually we skip them or treat as is.
            # The user says "gold_label is always string".
            # We will include them.
            conversations.append(format_conversation(item, system_prompt))

    return conversations


def load_mnli_dataset(
    train_path: Union[str, Path],
    system_prompt_path: Union[str, Path],
    test_path: Union[str, Path] = None,
):
    """
    Load MNLI dataset from JSONL files.

    Args:
        train_path: Path to the training JSONL file
        system_prompt_path: Path to the system prompt file
        test_path: Optional path to the test/validation JSONL file

    Returns:
        List of conversations or tuple of (train_conversations, test_conversations). Type:
        Union[List[List[Dict[str, str]]],
              Tuple[List[List[Dict[str, str]]],
                    List[List[Dict[str, str]]]]]
    """
    train_path = Path(train_path)
    system_prompt = load_system_prompt(system_prompt_path)

    train_data = process_jsonl(train_path, system_prompt)

    if test_path:
        test_path = Path(test_path)
        test_data = process_jsonl(test_path, system_prompt)
        return train_data, test_data

    return train_data
