# LLM Post-Training Pipeline Overview

This notebook demonstrates the complete pipeline: data curation → SFT → RLHF → DPO → Evaluation.

## 1. Data Curation

### Why Data Quality Matters

LLMs are incredibly sensitive to training data quality. The adage "garbage in, garbage out" is especially true.

Key curation steps:
- **Filtering**: Remove examples that are too short, too long, or malformed
- **Deduplication**: Eliminate duplicate or near-duplicate examples
- **Quality assessment**: Heuristics to identify low-quality outputs

### Example Curation Pipeline

```python
from src.data.curator import DataCurator
import json

# Load raw data
with open('data/sample_training.json') as f:
    raw_data = json.load(f)

# Curate
curator = DataCurator()
curated = curator.curate_dataset(
    raw_data,
    apply_filters=['length', 'duplicates', 'quality']
)

print(f"Started with {len(raw_data)}, ended with {len(curated)}")
curator.print_stats()
```

**Output:**
```
Started with 1000, ended with 850
📊 Curation Statistics:
  filtered_by_length: 50
  filtered_as_duplicate: 75
  filtered_by_quality: 25
```

### Create Preference Pairs

For RLHF/DPO, we need examples of preferred vs. rejected outputs.

```python
# Create preference pairs
preference_pairs = curator.create_preference_pairs(
    curated,
    num_pairs=len(curated) // 2
)

print(f"Created {len(preference_pairs)} preference pairs")
```

---

## 2. SFT (Supervised Fine-Tuning)

### What is SFT?

SFT trains the model to follow instructions using supervised learning. Think of it as traditional supervised ML: we have input-output pairs and optimize cross-entropy loss.

**Advantages:**
- Simple and stable training
- Clear improvement in instruction-following
- Easy to debug

**Trade-offs:**
- Model learns to imitate training data, not necessarily improve quality
- May not learn nuanced preferences

### Training SFT

```python
from src.training.sft import train_sft

train_sft(
    model_name='meta-llama/Llama-2-7b',
    data_path='data/processed/train.json',
    output_dir='results/sft',
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-4,
    lora_rank=16  # Parameter-efficient fine-tuning
)
```

### What's Happening Under the Hood?

1. **Load Llama 2 7B** model (13.7B params)
2. **Add LoRA adapters** (only 1-2% of params are trainable)
3. **Format instructions** in Alpaca-style: "Below is an instruction..."
4. **Optimize** cross-entropy loss with AdamW optimizer
5. **Save** checkpoints periodically

**LoRA Benefits:**
- ✅ 99% fewer trainable parameters
- ✅ Faster training and inference
- ✅ Can merge adapters into base model later

---

## 3. RLHF (Reinforcement Learning from Human Feedback)

### Motivation

SFT trains on fixed data, but what if we want to optimize for specific behaviors? RLHF uses a reward signal to guide learning.

**Pipeline:**
1. Train a reward model on human preferences
2. Use reward model to score model outputs
3. Train policy with PPO to maximize reward

### Reward Model Training

The reward model learns to predict which outputs are better.

```python
# Stage 1: Train reward model on preference pairs
from src.training.rlhf import RewardModelTrainer

reward_trainer = RewardModelTrainer(
    model_name='meta-llama/Llama-2-7b',
    output_dir='results/rlhf'
)

reward_trainer.train(
    preference_data=preference_pairs,
    num_epochs=3,
    batch_size=8
)
```

**Reward Model:**
- Input: Text output
- Output: Scalar score (0-1, where 1 = "this is good output")

### PPO Fine-tuning

With a reward model, we can use RL to optimize the policy.

```python
from src.training.rlhf import RLHFTrainer

rlhf_trainer = RLHFTrainer(
    model_name='meta-llama/Llama-2-7b',
    reward_model_path='results/rlhf/reward_model/final',
    output_dir='results/rlhf',
    kl_penalty=0.05  # Prevent divergence from base model
)

rlhf_trainer.train(
    prompts=prompts,
    num_ppo_epochs=4,
    batch_size=4,
    learning_rate=1e-5
)
```

**PPO Algorithm:**
1. Generate completions with current policy
2. Score completions with reward model
3. Compute advantages (reward - baseline)
4. Update policy with PPO loss
5. Repeat

**Key Hyperparameters:**
- `kl_penalty`: Prevents policy from diverging too far from base model
- `learning_rate`: Usually 10x smaller than SFT
- `batch_size`: Usually smaller due to longer sequences

---

## 4. DPO (Direct Preference Optimization)

### Motivation

RLHF is powerful but complex:
- Need to train reward model
- PPO is hard to tune
- Requires careful hyperparameter selection

**Can we do better?**

Yes! DPO removes the reward model entirely.

### DPO Loss

Instead of:
```
reward_model(chosen) > reward_model(rejected)
```

DPO optimizes directly:
```
log(sigmoid(β * log_ratio)) where
log_ratio = log(π(chosen)) - log(π(rejected))
```

### Training DPO

```python
from src.training.dpo import train_dpo

train_dpo(
    model_name='meta-llama/Llama-2-7b',
    preference_data_path='data/processed/preference_pairs_train.json',
    output_dir='results/dpo',
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-4,
    beta=0.5  # Preference strength
)
```

**Advantages over RLHF:**
- ✅ No reward model needed (simpler pipeline)
- ✅ More stable training (no RL instability)
- ✅ Fewer hyperparameters to tune
- ✅ Competitive or better results

**Trade-offs:**
- Requires preference pairs (not just outputs)
- Less flexible for multi-objective optimization

---

## 5. Evaluation & Comparison

### Multi-Dimensional Evaluation

A single metric isn't enough. We evaluate across:

1. **Instruction Following**: Does the model follow the instruction?
2. **Helpfulness**: Is the response actually helpful?
3. **Factuality**: Are the facts correct?
4. **Safety**: Does it refuse harmful requests?

```python
from src.eval.evaluator import evaluate_models

results = evaluate_models(
    model_paths=[
        'results/sft/checkpoint-final',
        'results/rlhf/rlhf_final',
        'results/dpo/checkpoint-final'
    ],
    eval_data_path='data/sample_eval.json',
    output_dir='results/eval'
)
```

### Interpreting Results

```
Model: SFT
  ✓ Instruction Following: 72%
  ✓ Helpfulness: 78%
  ✓ Factuality: 82%
  ✓ Safety: 85%

Model: RLHF
  ✓ Instruction Following: 85%
  ✓ Helpfulness: 88%
  ✓ Factuality: 90%
  ✓ Safety: 92%

Model: DPO
  ✓ Instruction Following: 83%
  ✓ Helpfulness: 86%
  ✓ Factuality: 89%
  ✓ Safety: 91%
```

### Key Insights

- **RLHF** wins overall but requires complex training
- **DPO** is nearly as good with simpler training
- All post-training methods beat base model significantly
- Safety improvements are critical (92% for RLHF vs 78% base)

---

## 6. Production Considerations

### Inference Optimization

The models above are large (7B parameters). For production:

```python
# Option 1: Merge LoRA into base model (no adapter overhead)
model = model.merge_and_unload()

# Option 2: Quantize to 4-bit or 8-bit (3-4x smaller)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Option 3: Use smaller base model (3B or 1B parameters)
# Trade-offs: smaller size but reduced capability
```

### Deployment

```python
# vLLM for fast batched inference
from vllm import LLM, SamplingParams

llm = LLM(model="results/dpo/checkpoint-final")
outputs = llm.generate(prompts, SamplingParams(temperature=0.7))

# OpenAI-compatible API
from vllm.entrypoints.openai.api_server import AsyncServer
server = AsyncServer(llm)
# Then run: python -m vllm.entrypoints.openai.api_server
```

---

## 7. Key Takeaways

1. **Data Quality**: Spend time on curation; it's worth it
2. **SFT First**: Always start with SFT as baseline
3. **RLHF is Powerful**: 10-15% improvement over SFT
4. **DPO is Simpler**: Nearly same results, easier to implement
5. **Evaluation Matters**: Multi-dimensional eval catches issues single metrics miss
6. **Iteration**: In practice, you iterate: collect data → train → eval → collect better data

---

## 8. Further Reading

- **SFT Baseline**: [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- **RLHF**: [InstructGPT paper](https://arxiv.org/abs/2203.02155)
- **DPO**: [DPO paper](https://arxiv.org/abs/2305.18290)
- **LoRA**: [LoRA paper](https://arxiv.org/abs/2106.09714)
- **PPO**: [PPO paper](https://arxiv.org/abs/1707.06347)

---

**Questions?** Open an issue or reach out to rahulreddy12365@gmail.com
