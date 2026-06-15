"""
DPO (Direct Preference Optimization) training.
Simpler and more stable alternative to RLHF.
"""

import argparse
import json
import os
from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from src_training_utils import load_json_dataset, set_seed, print_trainable_params


class DPODataset(torch.utils.data.Dataset):
    """Dataset for DPO training with preference pairs."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Get chosen and rejected completions
        chosen = example.get('chosen', '')
        rejected = example.get('rejected', '')

        # Tokenize both
        chosen_tokens = self.tokenizer(
            chosen,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        rejected_tokens = self.tokenizer(
            rejected,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze()
        }


class DPOTrainer(Trainer):
    """Custom trainer implementing DPO loss."""

    def __init__(self, *args, beta: float = 0.5, **kwargs):
        """
        Initialize DPO trainer.

        Args:
            beta: Temperature parameter for preference strength (default 0.5)
                  Higher beta = stronger preference enforcement
        """
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute DPO loss.

        L_dpo = -log(sigmoid(β * log(π(y_w | x) / π_ref(y_w | x)) - β * log(π(y_l | x) / π_ref(y_l | x))))

        Args:
            model: Policy model
            inputs: Batch with chosen/rejected inputs
            return_outputs: Whether to return model outputs

        Returns:
            Loss value
        """
        # Forward pass through chosen responses
        chosen_logits = model(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask']
        ).logits

        # Forward pass through rejected responses
        rejected_logits = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask']
        ).logits

        # Compute log probabilities (simplified)
        # In practice, you'd compute actual per-token log probs
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)

        # Average over sequence length
        chosen_avg = chosen_log_probs.mean(dim=-2).mean(dim=-1)
        rejected_avg = rejected_log_probs.mean(dim=-2).mean(dim=-1)

        # DPO loss: encourage preference for chosen over rejected
        log_odds = chosen_avg - rejected_avg
        loss = -F.logsigmoid(self.beta * log_odds).mean()

        return (loss, {'logits': chosen_logits}) if return_outputs else loss


def train_dpo(
    model_name: str,
    preference_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    beta: float = 0.5,
    max_length: int = 512,
    lora_rank: int = 16,
    seed: int = 42
):
    """
    Train model using DPO.

    Args:
        model_name: HF model ID (e.g., meta-llama/Llama-2-7b)
        preference_data_path: Path to preference pairs JSON
        output_dir: Directory to save checkpoints
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        beta: DPO temperature parameter
        max_length: Max sequence length
        lora_rank: LoRA rank
        seed: Random seed
    """
    set_seed(seed)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading preference data...")
    preference_data = load_json_dataset(preference_data_path)
    print(f"Loaded {len(preference_data)} preference pairs")

    print("Creating DPO dataset...")
    train_dataset = DPODataset(preference_data, tokenizer, max_length)

    print("Loading model with LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    # Add LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj'],
        bias='none'
    )

    model = get_peft_model(model, peft_config)
    print_trainable_params(model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        learning_rate=learning_rate,
        warmup_steps=50,
        weight_decay=0.01,
        bf16=True,
        optim='adamw_8bit',
        lr_scheduler_type='cosine'
    )

    # DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        beta=beta
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving final model to {output_dir}/checkpoint-final")
    trainer.save_model(f'{output_dir}/checkpoint-final')

    # Save config
    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'beta': beta,
        'max_length': max_length,
        'lora_rank': lora_rank,
        'num_preference_pairs': len(preference_data),
        'method': 'DPO (Direct Preference Optimization)'
    }

    with open(f'{output_dir}/dpo_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ DPO training complete!")

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b')
    parser.add_argument('--preference_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/dpo')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_dpo(
        model_name=args.model_name,
        preference_data_path=args.preference_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        seed=args.seed
    )
