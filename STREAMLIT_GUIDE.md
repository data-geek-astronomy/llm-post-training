# 📱 Streamlit Dashboard User Guide

Complete guide to using the interactive LLM post-training dashboard.

## 🎯 Dashboard Overview

The dashboard has 5 main sections:

```
🏠 Home          → Overview & quick start
📊 Data Curation → Upload & filter training data
⚙️ Training       → Configure & train models
🧪 Inference     → Test model outputs
📈 Results       → View metrics & compare
```

## 🏠 Home Page

**What it does**: Introduces the dashboard and explains each training method.

### Three Training Methods Explained

**SFT (Supervised Fine-Tuning)**
- Simple instruction-following training
- Good baseline
- Fast (15 min per 1K examples)
- Results: 30-50% improvement

**RLHF (Reinforcement Learning from Human Feedback)**
- Uses reward model to optimize quality
- Best results
- Slower (30 min per 1K examples)
- Results: 50-70% improvement

**DPO (Direct Preference Optimization)**
- Simpler alternative to RLHF
- No reward model needed
- Medium speed (20 min per 1K examples)
- Results: 45-65% improvement

### ⚡ Quick Start Links

Click "1. Upload Your Data" through "4. Compare Results" to follow the workflow.

---

## 📊 Data Curation

**Purpose**: Prepare and clean your training data.

### Tab 1: Upload

#### Option A: Upload Your Own Data

1. Click "Browse files" button
2. Select a JSON file with this format:

```json
[
  {
    "instruction": "What is machine learning?",
    "input": "",
    "output": "Machine learning allows computers..."
  },
  {
    "instruction": "Explain overfitting",
    "input": "",
    "output": "Overfitting occurs when..."
  }
]
```

3. File loads automatically
4. Preview shows first 2 examples

#### Option B: Use Sample Data

Click "Use Sample Data" button to load built-in examples.

### Tab 2: Preview & Filter

**Shows**:
- Total number of examples
- Average instruction length
- Average output length

**Filtering Options**:
- Min/Max instruction length (in tokens)
- Min/Max output length (in tokens)

**How to use**:
1. Adjust sliders to filter bad data
2. Click "Apply Filters"
3. See how many examples removed
4. Preview filtered data

### Tab 3: Export

**After filtering**:
1. Adjust train/test split ratio
2. Click "Download Train Data" or "Download Test Data"
3. Files download as JSON

**What this produces**:
- `train_data.json` - For training models
- `test_data.json` - For evaluation

---

## ⚙️ Training

**Purpose**: Configure and train models.

### Model Settings

**Base Model**
- `meta-llama/Llama-2-7b` (Default)
- `meta-llama/Llama-2-7b-chat` (Chat-optimized)

**Max Sequence Length**
- 256-1024 tokens
- Longer = more context but slower
- Recommended: 512

**LoRA Rank**
- 4, 8, 16, 32, 64
- Higher = more parameters, more VRAM
- Recommended: 16

### Training Settings

**Epochs**: How many times to train on data
- Recommended: 3
- More epochs = better results but risk overfitting

**Batch Size**: How many examples per training step
- 2, 4, 8, 16
- Larger = faster but needs more VRAM
- Recommended: 8

**Learning Rate**: How fast the model learns
- Too high = unstable
- Too low = slow training
- Recommended: 5e-4

### Select Training Method

Choose one of:
1. **SFT (Supervised Fine-Tuning)** - Recommended for first time
2. **RLHF (Reinforcement Learning)** - For best results
3. **DPO (Direct Preference)** - Fast alternative

### Deployment Mode

**🔴 Demo Mode (Recommended for Streamlit Cloud)**
- Uses pre-computed results
- No training happens
- Instant "results"
- Best for showcasing

**🟢 Local Training Mode**
- Actually trains models
- Requires GPU
- Takes 15-30 minutes
- Shows real results

### Start Training

1. Click "🚀 Start Training" button
2. Watch progress bar
3. Results appear when done

---

## 🧪 Inference (Testing)

**Purpose**: Generate predictions and test models.

### Tab 1: Single Prompt

**How to use**:

1. Enter a prompt/instruction in the text box
2. Adjust temperature (creativity):
   - Low (0.0) = Predictable
   - High (1.0) = Creative
3. Set max tokens (length of response)
4. Click "🚀 Generate Responses"

**What you get**:
- Outputs from SFT, RLHF, DPO models side-by-side
- Length in tokens for each
- Quality rating (★ stars)
- Thumbs up button to save preference

**Example Prompts**:
- "What is machine learning?"
- "Explain neural networks"
- "What are transformers?"
- "How does RLHF work?"

### Tab 2: Batch Test

**How to use**:

1. Enter multiple prompts (one per line)
2. Click "Evaluate"
3. See results table

**What you get**:
- Table showing all prompts
- ✅ checkmarks for successful generations
- Can compare across models

**Example**:
```
What is overfitting?
Explain backpropagation
What are hyperparameters?
```

---

## 📈 Results & Comparison

**Purpose**: View metrics and compare models.

### Tab 1: Metrics

Shows evaluation scores for all models:

| Metric | Meaning |
|--------|---------|
| Instruction Following | Does it follow instructions? |
| Helpfulness | Is the response helpful? |
| Factuality | Are facts correct? |
| Safety | Does it refuse harmful requests? |

**Comparison**:
```
SFT:  72% helpfulness
RLHF: 88% helpfulness (+16%)
DPO:  86% helpfulness (+14%)
```

### Tab 2: Comparison Chart

Visual bar chart comparing all metrics.

**How to read**:
- Y-axis: Score (0-100%)
- X-axis: Metric type
- Taller bars = better performance

### Tab 3: Analysis

**Key findings**:
- Which method wins
- Trade-offs between methods
- Recommendations

**Table shows**:
- Training time required
- GPU memory (VRAM) needed
- Expected results
- Implementation complexity

---

## 💡 Tips & Best Practices

### Data Preparation

✅ **Do**:
- Include diverse instructions
- Use clear, specific instructions
- Vary instruction length
- Provide detailed outputs

❌ **Don't**:
- Mix languages
- Use very short outputs (<10 tokens)
- Duplicate examples
- Include malformed data

### Training

✅ **Do**:
- Start with SFT (simpler)
- Use sample data first (test the flow)
- Monitor loss curves
- Save checkpoints

❌ **Don't**:
- Train with bad data
- Use batch size > GPU memory
- Train for too many epochs (overfitting)
- Use inconsistent instruction format

### Inference

✅ **Do**:
- Test on diverse prompts
- Compare all 3 methods
- Use temperature 0.5-0.8
- Save good outputs

❌ **Don't**:
- Only test one prompt
- Use temperature 0 (too boring)
- Use temperature > 1.0 (too random)
- Test on in-distribution examples only

---

## ❓ FAQ

### Q: Can I train on Streamlit Cloud?
**A**: Not really. Streamlit Cloud has no GPU and limited RAM. Use demo mode instead, or deploy locally/on cloud GPU.

### Q: How do I use my GPU?
**A**: Run locally:
```bash
streamlit run streamlit_app.py
```

Make sure PyTorch sees your GPU:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Q: Why is training slow?
**A**: Depends on:
- Model size (7B = medium speed)
- Batch size (larger = faster but more VRAM)
- Hardware (GPU >> CPU)
- Data size (more data = longer)

### Q: How do I upload large datasets?
**A**: For >10MB files:
1. Run locally instead of cloud
2. Or split data into chunks
3. Or provide URL to dataset

### Q: Can I use other models?
**A**: Yes, change in Training page:
- Llama 2 7B (current)
- Mistral 7B
- Qwen 7B
- Any HuggingFace model

### Q: How do I save trained models?
**A**: Models save to `results/` folder:
- `results/sft/checkpoint-final/` - SFT model
- `results/rlhf/rlhf_final/` - RLHF model
- `results/dpo/checkpoint-final/` - DPO model

Download the whole folder to save locally.

---

## 🎨 Customization

### Change Theme

Edit `.streamlit_config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"  # Your color
backgroundColor = "#F0F0F0"
textColor = "#1F1F1F"
font = "serif"
```

### Add Custom Instructions

Edit `streamlit_app.py` at line ~50 to change welcome message.

### Modify Metrics

Edit inference page (line ~400) to add/change evaluation metrics.

---

## 🆘 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements_streamlit.txt
```

### "CUDA out of memory"
- Reduce batch size (2 instead of 8)
- Reduce max sequence length (256)
- Use smaller model

### "Model not found"
```bash
huggingface-cli login
# Then enter your HuggingFace token
```

### App crashes
Check error message in terminal:
```bash
streamlit run streamlit_app.py
# Look at the error output
```

### Slow inference
- Use smaller model
- Reduce max tokens
- Use GPU instead of CPU

---

## 📞 Support

Need help?
- **Email**: rahulreddy12365@gmail.com
- **GitHub**: https://github.com/yourusername/llm-post-training
- **Discord**: [Link to community server]

---

**Happy training! 🚀**
