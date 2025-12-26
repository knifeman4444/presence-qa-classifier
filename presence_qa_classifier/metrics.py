from typing import List, Optional

def calculate_accuracy(predictions: List[str], references: List[str], filter_labels: Optional[List[str]] = None) -> float:
    """
    Calculate accuracy score between predictions and references.
    Optionally filters by reference labels.
    
    Args:
        predictions: List of predicted labels/strings
        references: List of ground truth labels/strings
        filter_labels: If provided, only calculate accuracy for samples where reference is in this list.
        
    Returns:
        Accuracy score as a float
    """
    if not predictions or not references:
        return 0.0
    
    if len(predictions) != len(references):
        raise ValueError(f"Predictions and references must have the same length. Got {len(predictions)} and {len(references)}")

    correct_count = 0
    total_count = 0

    for pred, ref in zip(predictions, references):
        if filter_labels is not None:
            if ref not in filter_labels:
                continue
        
        if pred == ref:
            correct_count += 1
        total_count += 1
        
    if total_count == 0:
        return 0.0
            
    return float(correct_count / total_count)
    