# Quick Start Guide

Get the LLM post-training pipeline running in 30 minutes (with sample data).

## 1. Install & Setup

```bash
# Clone repo
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training

# Install dependencies
pip install -r requirements.txt

# Optional: Set up Weights & Biases for logging
wandb login
```

## 2. Prepare Data

```bash
# Curate sample dataset
python src/data/curator.py \
  --input_path data/sample_training.json \
  --output_dir data/processed \
  --num_train 100 \
  --num_test 20
```

This creates:
- `data/processed/train.json` - Training examples
- `data/processed/test.json` - Test examples
- `data/processed/preference_pairs_train.json` - For RLHF/DPO
- `data/processed/preference_pairs_test.json`

## 3. Run SFT Training

```bash
python src/training/sft.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --data_path data/processed/train.json \
  --output_dir results/sft \
  --num_epochs 1 \
  --batch_size 4
```

**Output**: `results/sft/checkpoint-final/` with trained weights

## 4. Run RLHF

```bash
python src/training/rlhf.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --preference_data_path data/processed/preference_pairs_train.json \
  --prompt_data_path data/processed/train.json \
  --output_dir results/rlhf \
  --num_reward_epochs 1 \
  --num_ppo_epochs 1
```

**Output**: Reward model + RLHF checkpoint

## 5. Run DPO

```bash
python src/training/dpo.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --preference_data_path data/processed/preference_pairs_train.json \
  --output_dir results/dpo \
  --num_epochs 1 \
  --batch_size 4
```

**Output**: `results/dpo/checkpoint-final/`

## 6. Evaluate & Compare

```bash
python src/eval/evaluator.py \
  --models results/sft/checkpoint-final results/rlhf/rlhf_final results/dpo/checkpoint-final \
  --eval_data data/sample_eval.json \
  --output_dir results/eval
```

**Output**: `results/eval/evaluation_results.json` with metrics for all 3 models

## 7. View Results

```bash
# Check evaluation results
cat results/eval/evaluation_results.json | jq .

# View training logs
tail -f results/sft/training_args.bin
```

## 🎯 What You'll Learn

- ✅ **Data curation**: Filtering, deduplication, quality checks
- ✅ **SFT**: Fine-tuning with LoRA for efficiency
- ✅ **RLHF**: Reward model + PPO pipeline
- ✅ **DPO**: Simpler preference optimization
- ✅ **Evaluation**: Multi-dimensional metrics (helpfulness, factuality, safety)

## 🚀 Next Steps

1. **Use your own data**: Replace `data/sample_training.json` with your dataset
2. **Tune hyperparameters**: Adjust learning rates, batch sizes, epochs
3. **Extend evaluation**: Add more metrics or domain-specific eval sets
4. **Deploy**: Convert checkpoints to ONNX or quantize for inference

## 📊 Expected Training Times (on single GPU)

- **SFT** (100 examples): 2-5 minutes
- **RLHF** (100 preference pairs): 5-10 minutes
- **DPO** (100 preference pairs): 3-8 minutes
- **Evaluation**: 2-3 minutes

*Times vary with GPU memory, batch size, and sequence length.*

## 🐛 Troubleshooting

**CUDA out of memory?**
- Reduce `batch_size` or `max_length`
- Enable `bfloat16` quantization

**Models not downloading?**
- Check HF auth: `huggingface-cli login`
- Ensure internet connection for model download

**Evaluation hangs?**
- Reduce number of eval examples in evaluator
- Check GPU availability

## 📚 Full Documentation

See README.md for detailed architecture, methodology, and references.

---

Questions? Open an issue or email rahulreddy12365@gmail.com
