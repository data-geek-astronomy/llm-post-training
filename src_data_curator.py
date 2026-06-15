"""
Data curation pipeline for instruction-following datasets.
Handles filtering, deduplication, and quality checks.
"""

import json
import os
from typing import List, Dict
import numpy as np
from collections import defaultdict


class DataCurator:
    """Curate instruction-following datasets."""

    def __init__(self):
        self.stats = defaultdict(int)

    def filter_by_length(
        self,
        examples: List[Dict],
        min_instruction_len: int = 10,
        max_instruction_len: int = 500,
        min_output_len: int = 10,
        max_output_len: int = 2000
    ) -> List[Dict]:
        """Remove examples with invalid lengths."""
        filtered = []

        for ex in examples:
            inst_len = len(ex.get('instruction', '').split())
            out_len = len(ex.get('output', '').split())

            if (min_instruction_len <= inst_len <= max_instruction_len and
                min_output_len <= out_len <= max_output_len):
                filtered.append(ex)
            else:
                self.stats['filtered_by_length'] += 1

        return filtered

    def remove_duplicates(
        self,
        examples: List[Dict]
    ) -> List[Dict]:
        """Remove duplicate examples."""
        seen = set()
        filtered = []

        for ex in examples:
            # Use instruction + output as unique key
            key = (ex.get('instruction', '').lower().strip(),
                   ex.get('output', '').lower().strip())

            if key not in seen:
                filtered.append(ex)
                seen.add(key)
            else:
                self.stats['filtered_as_duplicate'] += 1

        return filtered

    def filter_by_quality(
        self,
        examples: List[Dict],
        min_output_quality: float = 0.5
    ) -> List[Dict]:
        """
        Filter by output quality heuristics.

        Checks:
        - Output is not just repeated characters
        - Output has reasonable diversity
        - No excessive punctuation
        """
        filtered = []

        for ex in examples:
            output = ex.get('output', '')

            # Check for repeated characters
            if self._has_repeated_chars(output):
                self.stats['filtered_by_repeated_chars'] += 1
                continue

            # Check for excessive punctuation
            if self._has_excessive_punctuation(output):
                self.stats['filtered_by_excessive_punct'] += 1
                continue

            # Check for meaningfulness
            if self._is_meaningful(output):
                filtered.append(ex)
            else:
                self.stats['filtered_by_low_quality'] += 1

        return filtered

    def _has_repeated_chars(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text has too many repeated characters."""
        if not text:
            return True

        # Count unique characters
        unique_chars = len(set(text))
        char_diversity = unique_chars / len(text)

        return char_diversity < threshold

    def _has_excessive_punctuation(self, text: str, threshold: float = 0.5) -> bool:
        """Check for excessive punctuation."""
        if not text:
            return True

        punct_chars = sum(1 for c in text if c in '!?.,;:')
        punct_ratio = punct_chars / len(text)

        return punct_ratio > threshold

    def _is_meaningful(self, text: str) -> bool:
        """Check if text is meaningful (not gibberish)."""
        if len(text) < 5:
            return False

        # Simple check: has actual words (not just symbols)
        words = text.split()
        if not words:
            return False

        # At least some word-like tokens
        return len(words) > 2

    def create_preference_pairs(
        self,
        base_examples: List[Dict],
        num_pairs: int = None
    ) -> List[Dict]:
        """
        Create preference pairs by selecting good vs. lower-quality outputs.

        In practice, you'd use human annotations or model comparisons.
        This is a simplified heuristic: longer, more detailed outputs are "chosen".
        """
        if num_pairs is None:
            num_pairs = len(base_examples) // 2

        pairs = []

        for i in range(min(num_pairs, len(base_examples) - 1)):
            ex1 = base_examples[i]
            ex2 = base_examples[i + 1]

            # Create preference pair based on output length (heuristic)
            output1_len = len(ex1.get('output', '').split())
            output2_len = len(ex2.get('output', '').split())

            if output1_len > output2_len:
                chosen = ex1
                rejected = ex2
            else:
                chosen = ex2
                rejected = ex1

            pair = {
                'instruction': chosen.get('instruction', ''),
                'chosen': self._format_example(chosen),
                'rejected': self._format_example(rejected)
            }

            pairs.append(pair)

        self.stats['preference_pairs_created'] = len(pairs)
        return pairs

    def _format_example(self, ex: Dict) -> str:
        """Format example as text."""
        instruction = ex.get('instruction', '')
        input_text = ex.get('input', '')
        output = ex.get('output', '')

        text = f"Instruction: {instruction}\n"
        if input_text:
            text += f"Input: {input_text}\n"
        text += f"Response: {output}"

        return text

    def curate_dataset(
        self,
        examples: List[Dict],
        apply_filters: List[str] = None
    ) -> List[Dict]:
        """
        Apply full curation pipeline.

        Args:
            examples: Raw examples
            apply_filters: List of filters to apply
                          (default: all filters)

        Returns:
            Curated examples
        """
        if apply_filters is None:
            apply_filters = ['length', 'duplicates', 'quality']

        curated = examples.copy()
        print(f"Starting with {len(curated)} examples")

        if 'length' in apply_filters:
            print("Filtering by length...")
            curated = self.filter_by_length(curated)
            print(f"  → {len(curated)} examples remain")

        if 'duplicates' in apply_filters:
            print("Removing duplicates...")
            curated = self.remove_duplicates(curated)
            print(f"  → {len(curated)} examples remain")

        if 'quality' in apply_filters:
            print("Filtering by quality...")
            curated = self.filter_by_quality(curated)
            print(f"  → {len(curated)} examples remain")

        return curated

    def print_stats(self):
        """Print curation statistics."""
        print("\n📊 Curation Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")


def main(
    input_path: str,
    output_dir: str,
    num_train_examples: int = 1000,
    num_test_examples: int = 200
):
    """
    Curate and split dataset.

    Args:
        input_path: Path to raw dataset (JSON)
        output_dir: Output directory
        num_train_examples: Number of training examples
        num_test_examples: Number of test examples
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    # Curate
    curator = DataCurator()
    curated = curator.curate_dataset(raw_data)

    curator.print_stats()

    # Create preference pairs
    print("\nCreating preference pairs...")
    preference_pairs = curator.create_preference_pairs(curated)

    # Split into train/test
    total_needed = num_train_examples + num_test_examples
    if len(curated) < total_needed:
        print(f"⚠️  Only {len(curated)} examples available, need {total_needed}")
        num_train_examples = int(len(curated) * 0.8)
        num_test_examples = len(curated) - num_train_examples

    train_data = curated[:num_train_examples]
    test_data = curated[num_train_examples:num_train_examples + num_test_examples]

    # Preference pairs split
    num_pref_train = int(len(preference_pairs) * 0.8)
    pref_train = preference_pairs[:num_pref_train]
    pref_test = preference_pairs[num_pref_train:]

    # Save splits
    print(f"\nSaving datasets to {output_dir}...")

    with open(f'{output_dir}/train.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(f'{output_dir}/test.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    with open(f'{output_dir}/preference_pairs_train.json', 'w') as f:
        json.dump(pref_train, f, indent=2)

    with open(f'{output_dir}/preference_pairs_test.json', 'w') as f:
        json.dump(pref_test, f, indent=2)

    print(f"✅ Dataset curation complete!")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    print(f"  Preference pairs (train): {len(pref_train)}")
    print(f"  Preference pairs (test): {len(pref_test)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=200)

    args = parser.parse_args()

    main(args.input_path, args.output_dir, args.num_train, args.num_test)
