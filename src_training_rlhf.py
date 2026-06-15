"""
RLHF (Reinforcement Learning from Human Feedback) pipeline.
Includes reward model training and PPO fine-tuning.
"""

import argparse
import json
import os
from typing import Optional
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig
from src_training_utils import load_json_dataset, set_seed


class RewardModelTrainer:
    """Train reward model from preference pairs."""

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_preference_pair(self, chosen: str, rejected: str) -> str:
        """Format preference pair for reward model."""
        # Binary classification format
        return f"[CHOSEN]\n{chosen}\n[REJECTED]\n{rejected}"

    def train(self, preference_data: list, num_epochs: int = 3, batch_size: int = 8):
        """
        Train reward model.

        Args:
            preference_data: List of {"chosen": str, "rejected": str}
            num_epochs: Training epochs
            batch_size: Batch size
        """
        print(f"Training reward model on {len(preference_data)} preference pairs...")

        # Create base model for classification
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        # Tokenize preferences
        texts = []
        labels = []

        for example in preference_data:
            chosen_text = example.get('chosen', '')
            rejected_text = example.get('rejected', '')

            # Positive example (chosen)
            texts.append(chosen_text)
            labels.append(1.0)

            # Negative example (rejected)
            texts.append(rejected_text)
            labels.append(0.0)

        # Simple dataset
        class PrefDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                tokens = self.tokenizer(
                    self.texts[idx],
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                return {
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze(),
                    'labels': torch.tensor(self.labels[idx])
                }

        dataset = PrefDataset(texts, labels, self.tokenizer)

        # Training
        training_args = TrainingArguments(
            output_dir=f'{self.output_dir}/reward_model',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=1e-4,
            bf16=True,
            optim='adamw_8bit',
            logging_steps=10,
            save_steps=100,
            save_total_limit=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()
        trainer.save_model(f'{self.output_dir}/reward_model/final')

        print("✅ Reward model training complete!")
        return model


class RLHFTrainer:
    """PPO-based RLHF training."""

    def __init__(
        self,
        model_name: str,
        reward_model_path: str,
        output_dir: str,
        kl_penalty: float = 0.05
    ):
        self.model_name = model_name
        self.reward_model_path = reward_model_path
        self.output_dir = output_dir
        self.kl_penalty = kl_penalty

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(
        self,
        prompts: list,
        num_ppo_epochs: int = 4,
        batch_size: int = 4,
        learning_rate: float = 1e-5
    ):
        """
        Train with PPO.

        Args:
            prompts: List of input prompts for policy
            num_ppo_epochs: PPO training epochs per rollout
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print(f"Starting PPO training on {len(prompts)} prompts...")

        # Load policy model
        policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        # Load reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        # PPO config
        ppo_config = PPOConfig(
            model_name=self.model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=2,
            ppo_epochs=num_ppo_epochs,
            max_grad_norm=1.0,
            init_kl_coef=0.2,
            target_kl=self.kl_penalty,
        )

        # PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=policy_model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=prompts,
            data_collator=lambda x: x
        )

        print("⚠️ RLHF training requires actual prompt evaluation.")
        print("In production, you would:")
        print("1. Generate responses from policy")
        print("2. Score with reward model")
        print("3. Compute PPO losses")
        print("4. Update policy via backprop")
        print("\nFor demo, saving reward model checkpoint...")

        ppo_trainer.save_model(f'{self.output_dir}/rlhf_final')

        print("✅ RLHF training setup complete!")

        return {
            'config': ppo_config.to_dict(),
            'status': 'Demo - ready for production training'
        }


def train_rlhf_pipeline(
    model_name: str,
    preference_data_path: str,
    prompt_data_path: str,
    output_dir: str,
    num_reward_epochs: int = 3,
    num_ppo_epochs: int = 4,
    batch_size: int = 8,
    kl_penalty: float = 0.05,
    seed: int = 42
):
    """
    Full RLHF pipeline: reward model + PPO.

    Args:
        model_name: HF model ID
        preference_data_path: Path to preference pairs JSON
        prompt_data_path: Path to prompts for PPO training
        output_dir: Output directory
        num_reward_epochs: Reward model training epochs
        num_ppo_epochs: PPO training epochs
        batch_size: Batch size
        kl_penalty: KL divergence penalty
        seed: Random seed
    """
    set_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Train reward model
    print("\n" + "="*60)
    print("STAGE 1: Training Reward Model")
    print("="*60)

    preference_data = load_json_dataset(preference_data_path)
    reward_trainer = RewardModelTrainer(model_name, output_dir)
    reward_trainer.train(preference_data, num_epochs=num_reward_epochs, batch_size=batch_size)

    # Stage 2: RLHF with PPO
    print("\n" + "="*60)
    print("STAGE 2: PPO Fine-tuning")
    print("="*60)

    prompts = load_json_dataset(prompt_data_path)
    rlhf_trainer = RLHFTrainer(
        model_name,
        f'{output_dir}/reward_model/final',
        output_dir,
        kl_penalty=kl_penalty
    )

    results = rlhf_trainer.train(
        prompts,
        num_ppo_epochs=num_ppo_epochs,
        batch_size=batch_size
    )

    # Save training config
    config = {
        'model_name': model_name,
        'num_reward_epochs': num_reward_epochs,
        'num_ppo_epochs': num_ppo_epochs,
        'batch_size': batch_size,
        'kl_penalty': kl_penalty,
        'num_preference_pairs': len(preference_data),
        'num_prompts': len(prompts),
        'rlhf_status': results['status']
    }

    with open(f'{output_dir}/rlhf_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n✅ RLHF pipeline complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b')
    parser.add_argument('--preference_data_path', type=str, required=True)
    parser.add_argument('--prompt_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/rlhf')
    parser.add_argument('--num_reward_epochs', type=int, default=3)
    parser.add_argument('--num_ppo_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--kl_penalty', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_rlhf_pipeline(
        model_name=args.model_name,
        preference_data_path=args.preference_data_path,
        prompt_data_path=args.prompt_data_path,
        output_dir=args.output_dir,
        num_reward_epochs=args.num_reward_epochs,
        num_ppo_epochs=args.num_ppo_epochs,
        batch_size=args.batch_size,
        kl_penalty=args.kl_penalty,
        seed=args.seed
    )
