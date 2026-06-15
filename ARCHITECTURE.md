# Architecture & Design Decisions

## Overview

This project implements a complete LLM post-training pipeline with three approaches (SFT, RLHF, DPO) on Llama 2 7B. The design prioritizes clarity, reproducibility, and production-readiness.

## Design Principles

1. **Modular**: Each training method is independent but shares utilities
2. **Reproducible**: Fixed seeds, documented configs, clear data flow
3. **Efficient**: LoRA reduces parameters; bfloat16 reduces memory
4. **Observable**: Logging, checkpoints, and metrics at every step
5. **Scalable**: Code works on single GPU but designed for multi-GPU training

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN PIPELINE                           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │   SFT    │  │  RLHF    │  │   DPO    │
        │Training  │  │Training  │  │Training  │
        └──────────┘  └──────────┘  └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Evaluation      │
                    │   Framework       │
                    └───────────────────┘
```

## Data Flow

### 1. Data Curation Pipeline

```
Raw Data (JSON)
    ↓
[Length Filter]      Remove too short/long examples
    ↓
[Deduplication]      Remove near-duplicates
    ↓
[Quality Filter]     Remove gibberish/low-quality
    ↓
Curated Data
    ├─→ Train Split (80%)
    ├─→ Test Split (20%)
    └─→ Preference Pairs
         ├─→ Preference Train (80%)
         └─→ Preference Test (20%)
```

**Key Classes:**
- `DataCurator`: Filtering, deduplication, quality checks
- `SFTDataset`: Tokenization for SFT training
- `DPODataset`: Preference pair tokenization

### 2. SFT Training Pipeline

```
Base Model (Llama 2 7B)
    ↓
[Add LoRA Adapters]      ~1% of parameters trainable
    ↓
SFT Dataset
    ├─→ Format as Alpaca-style instructions
    ├─→ Tokenize (max_length=512)
    └─→ Create labels = input_ids
    ↓
[HF Trainer]
    ├─→ Forward pass: compute cross-entropy loss
    ├─→ Backward pass: update LoRA weights
    └─→ Log metrics: loss, learning rate, throughput
    ↓
SFT Checkpoint
    └─→ Save adapter weights + config
```

**Key Components:**
- `create_lora_model()`: Adds LoRA to base model
- `SFTDataset`: Custom PyTorch dataset
- `Trainer`: HF Trainer with bfloat16 and AdamW 8-bit

**Hyperparameters:**
- `lora_rank=16`: LoRA rank (typically 8-32)
- `learning_rate=5e-4`: Smaller than pre-training
- `warmup_steps=100`: Linear warmup
- `lr_scheduler_type=cosine`: Cosine annealing

### 3. RLHF Training Pipeline

```
SFT Checkpoint
    ├─→ Reward Model Training
    │   ├─→ Load base model
    │   ├─→ Add classification head (1 scalar output)
    │   ├─→ Train on preference pairs
    │   │   ├─→ Chosen output: label = 1
    │   │   └─→ Rejected output: label = 0
    │   ├─→ Loss: binary cross-entropy
    │   └─→ Save reward model
    │
    └─→ PPO Fine-tuning
        ├─→ Load policy model (SFT)
        ├─→ Load reference model (SFT, frozen)
        ├─→ Generate rollouts
        │   ├─→ Sample prompts
        │   ├─→ Generate completions (policy)
        │   └─→ Score with reward model
        ├─→ Compute advantages (reward - baseline)
        ├─→ PPO loss = -clipped_advantage
        ├─→ KL penalty = β * KL(policy || reference)
        └─→ Save policy checkpoint
```

**Key Equations:**

Reward Model Loss:
```
L = BCE(reward_model(chosen), 1) + BCE(reward_model(rejected), 0)
```

PPO Loss:
```
L_ppo = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
L_total = L_ppo + β * KL(π || π_ref)
```

where:
- `r_t`: probability ratio
- `A_t`: advantage estimate
- `β`: KL coefficient

### 4. DPO Training Pipeline

```
Base Model (Llama 2 7B)
    ↓
[Add LoRA Adapters]
    ↓
Preference Pairs
    ├─→ Chosen: forward pass → log probs
    ├─→ Rejected: forward pass → log probs
    └─→ DPO Loss
        L = -log(sigmoid(β * (log π(chosen) - log π(rejected))))
    ↓
[Update LoRA weights]
    ↓
DPO Checkpoint
```

**Why DPO Works:**

Traditional RLHF:
```
maximize E[r(chosen) - r(rejected)]  with KL penalty
```

DPO directly optimizes:
```
maximize log π(chosen | x) - log π(rejected | x)
```

This is equivalent but simpler (no reward model needed).

### 5. Evaluation Framework

```
Model Checkpoints (SFT, RLHF, DPO)
    │
    ├─→ Instruction-Following Eval
    │   ├─→ Generate responses
    │   ├─→ Compare with expected outputs
    │   └─→ Metric: exact-match accuracy
    │
    ├─→ Helpfulness Eval
    │   ├─→ Generate responses
    │   ├─→ LLM-as-judge (GPT-4 in production)
    │   └─→ Metric: 1-5 helpfulness score
    │
    ├─→ Factuality Eval
    │   ├─→ Generate responses
    │   ├─→ Check against knowledge base
    │   └─→ Metric: % correct facts
    │
    └─→ Safety Eval
        ├─→ Test on harmful prompts
        ├─→ Check if model refuses
        └─→ Metric: % safe refusals

Results JSON
    └─→ Comparison plots
```

## Key Design Decisions

### 1. LoRA for Parameter Efficiency

**Decision**: Use LoRA instead of full fine-tuning

**Rationale:**
- Llama 2 7B has 7B parameters; full tuning = 28GB memory (fp32)
- LoRA reduces trainable params to ~0.1% (20M params)
- Training memory drops from 28GB to ~4GB
- No accuracy loss; comparable results to full fine-tuning

**Implementation:**
```python
from peft import get_peft_model, LoraConfig
config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # scaling
    target_modules=['q_proj', 'v_proj'],  # which layers
)
model = get_peft_model(model, config)
```

### 2. bfloat16 for Training

**Decision**: Use bfloat16 instead of float32

**Rationale:**
- bfloat16: 16-bit, wider range than float16
- 2x memory savings (1 param = 2 bytes instead of 4)
- Stable training without loss scaling
- Supported by modern GPUs (A100, H100, RTX 40x)

### 3. AdamW 8-bit Optimizer

**Decision**: Use 8-bit AdamW (via BitsAndBytes)

**Rationale:**
- Optimizer states normally 8x the model size
- 8-bit optimizer states = 4x model size
- Total training memory: 8GB instead of 28GB
- No convergence issues

### 4. Alpaca Format for Instructions

**Decision**: Use Alpaca-style formatting

**Rationale:**
- Industry standard (widely used in open-source projects)
- Clear structure (Instruction/Input/Response)
- Easy to understand and modify
- Generalizes well to different instruction types

### 5. Separate Evaluation Script

**Decision**: Evaluation separate from training

**Rationale:**
- Decoupled concerns (training vs. eval)
- Can evaluate without retraining
- Easy to add new metrics
- Reproducible eval on fixed set

## Training Hyperparameters

### SFT

```yaml
model: meta-llama/Llama-2-7b
optimizer: adamw_8bit
learning_rate: 5e-4
warmup_steps: 100
lr_scheduler: cosine
batch_size: 8
max_length: 512
lora_rank: 16
epochs: 3
```

### RLHF

```yaml
# Reward Model
reward_epochs: 3
reward_lr: 1e-4
reward_batch_size: 8

# PPO
ppo_epochs: 4
ppo_lr: 1e-5
ppo_batch_size: 4
kl_penalty: 0.05  # β in KL divergence
max_grad_norm: 1.0
```

### DPO

```yaml
model: meta-llama/Llama-2-7b
optimizer: adamw_8bit
learning_rate: 5e-4
warmup_steps: 50
lr_scheduler: cosine
batch_size: 8
max_length: 512
lora_rank: 16
epochs: 3
beta: 0.5  # preference strength
```

## Reproducibility

### Fixed Seeds

```python
from src.training.utils import set_seed
set_seed(42)
```

Ensures:
- Same random initialization
- Same data shuffling
- Same dropout patterns

### Deterministic Algorithms

```python
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
```

### Config Logging

Each training run saves:
- `sft_config.json` / `rlhf_config.json` / `dpo_config.json`
- Training arguments
- Data statistics
- Hyperparameters

### Versioning

Training configs include:
- Model version (e.g., Llama-2-7b-hf)
- Training date/time
- Number of examples used
- Seed value

## Scalability Considerations

### Current Setup (Single GPU)

- Max batch size: 8 (per GPU)
- Max sequence length: 512
- Training time: ~30 min for 1K examples (per method)

### Multi-GPU (DDP)

```python
from torch.nn.parallel import DataParallel
model = DataParallel(model, device_ids=[0, 1, 2, 3])
```

Enables:
- Larger batch sizes (8 per GPU × 4 GPUs = 32)
- Faster training (4x wall-clock time)

### Distributed (Multi-node)

Use HF Accelerate:
```bash
accelerate launch --multi_gpu src/training/sft.py ...
```

## Testing Strategy

### Unit Tests (Future)

```python
def test_data_curator():
    examples = [...]
    curator = DataCurator()
    curated = curator.filter_by_length(examples)
    assert len(curated) < len(examples)
```

### Integration Tests (Future)

```python
def test_sft_training():
    # Train on small dataset, verify loss decreases
    ...
```

### Regression Tests (Future)

Track eval metrics over time to detect degradation.

## Future Improvements

1. **Multi-GPU Training**: Implement DDP for faster training
2. **Quantization**: Support 4-bit/8-bit inference
3. **Merging**: Auto-merge LoRA weights into base model
4. **Fine-grained Control**: Per-layer LoRA rank, selective unfreezing
5. **Advanced Evals**: GPT-4 judging, BLEU/ROUGE, human evals
6. **Prompt Engineering**: Systematic evaluation of prompt templates
7. **Synthetic Data**: Use model to generate preference pairs

## References

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09714)
- [InstructGPT: RLHF at Scale](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Llama 2: Open Foundation & Fine-Tuned Models](https://arxiv.org/abs/2307.09288)

---

**Questions about architecture?** Open an issue or email rahulreddy12365@gmail.com
