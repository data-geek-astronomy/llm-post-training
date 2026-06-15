"""
Supervised Fine-Tuning (SFT) pipeline for instruction-following.
"""

import argparse
import json
import os
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from src_training_utils import (
    load_json_dataset,
    tokenize_dataset,
    set_seed,
    print_trainable_params
)


class SFTDataset(torch.utils.data.Dataset):
    """Dataset for SFT training."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Format instruction
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')

        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        prompt += f"### Instruction:\n{instruction}\n"

        if input_text:
            prompt += f"### Input:\n{input_text}\n"

        prompt += f"### Response:\n{output_text}"

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze().clone()
        }


def create_lora_model(model_name: str, lora_rank: int = 16, lora_alpha: int = 32):
    """Create base model with LoRA adapters."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
        inference_mode=False
    )

    model = get_peft_model(model, peft_config)
    return model


def train_sft(
    model_name: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    max_length: int = 512,
    lora_rank: int = 16,
    seed: int = 42
):
    """
    Train model with SFT.

    Args:
        model_name: HF model ID (e.g., meta-llama/Llama-2-7b)
        data_path: Path to training data (JSON)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        lora_rank: LoRA rank for parameter efficiency
        seed: Random seed
    """
    set_seed(seed)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")
    data = load_json_dataset(data_path)
    print(f"Loaded {len(data)} training examples")

    print("Creating dataset...")
    train_dataset = SFTDataset(data, tokenizer, max_length)

    print("Creating model with LoRA...")
    model = create_lora_model(model_name, lora_rank=lora_rank)
    print_trainable_params(model)

    # Training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        optim='adamw_8bit',
        lr_scheduler_type='cosine'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {output_dir}/checkpoint-final")
    trainer.save_model(f'{output_dir}/checkpoint-final')

    # Save training config
    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'lora_rank': lora_rank,
        'num_examples': len(data)
    }
    with open(f'{output_dir}/sft_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ SFT training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/sft')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_sft(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        seed=args.seed
    )
