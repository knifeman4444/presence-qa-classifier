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

    passage = item.get("passage", "")
    question = item.get("question", "")
    answer = item.get("answer")

    # Format user input with system prompt
    user_content = f'{system_prompt}\n\nPassage: "{passage}"\nQuestion: "{question}"'

    # Format assistant answer
    if answer is True:
        assistant_content = "yes"
    elif answer is False:
        assistant_content = "no"
    elif answer == "no-answer":
        assistant_content = "idk"
    else:
        raise ValueError(f"Unknown answer: {answer}")

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def process_jsonl(file_path: Path, system_prompt: str) -> List[List[Dict[str, str]]]:
    """Read JSONL file and convert to list of conversations."""
    conversations = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            conversations.append(format_conversation(item, system_prompt))

    return conversations


def load_boolq_dataset(
    train_path: Union[str, Path],
    system_prompt_path: Union[str, Path],
    test_path: Union[str, Path] = None,
):
    """
    Load BoolQ dataset from JSONL files.

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
