"""
Common utilities for training pipeline.
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer


def load_json_dataset(path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json_dataset(data: List[Dict], path: str) -> None:
    """Save dataset to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def format_instruction(example: Dict) -> str:
    """Format instruction-following example for SFT."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')

    # Alpaca-style formatting
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    prompt += f"### Instruction:\n{instruction}\n"

    if input_text:
        prompt += f"### Input:\n{input_text}\n"

    prompt += f"### Response:\n{output_text}"

    return prompt


def tokenize_dataset(
    dataset: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    is_preference: bool = False
) -> Dict:
    """
    Tokenize dataset for training.

    Args:
        dataset: List of examples
        tokenizer: HF tokenizer
        max_length: Max sequence length
        is_preference: If True, expects 'chosen' and 'rejected' keys

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and optional 'labels'
    """
    if is_preference:
        # For preference pairs (RLHF/DPO)
        chosen_texts = [format_instruction(ex['chosen']) for ex in dataset]
        rejected_texts = [format_instruction(ex['rejected']) for ex in dataset]

        chosen_tokens = tokenizer(
            chosen_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        rejected_tokens = tokenizer(
            rejected_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        return {
            'chosen': chosen_tokens,
            'rejected': rejected_tokens
        }
    else:
        # For SFT
        texts = [format_instruction(ex) for ex in dataset]

        tokens = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        # Set labels (same as input_ids for causal LM loss)
        tokens['labels'] = tokens['input_ids'].clone()

        return tokens


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_size(model) -> int:
    """Get model size in millions of parameters."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def print_trainable_params(model) -> None:
    """Print trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable / 1e6:.1f}M")
    print(f"Total params: {total / 1e6:.1f}M")
    print(f"Trainable%: {100 * trainable / total:.1f}%")
