# 🎨 Interactive Streamlit Dashboard

**Live Demo** | **Data Curation** | **Model Training** | **Inference Testing** | **Results Comparison**

Transform your LLM post-training project into an interactive web app with just one command.

## 🚀 Quick Start (2 minutes)

### Local Deployment

```bash
# 1. Install dependencies
pip install -r requirements_streamlit.txt

# 2. Run the app
streamlit run streamlit_app.py

# 3. Opens at http://localhost:8501
```

That's it! 🎉

### Cloud Deployment (Streamlit Cloud - Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Select your repo → Select `streamlit_app.py`
4. App deploys automatically (live in 2-3 minutes)

Your app URL: `https://<username>-llm-post-training.streamlit.app`

## 📱 Dashboard Features

### 🏠 Home Page
- Project overview
- Training method explanations
- Quick start guide
- Links to documentation

### 📊 Data Curation
- **Upload Tab**: Load your JSON training data or use samples
- **Filter Tab**: Remove duplicates, filter by length/quality
- **Export Tab**: Download train/test splits

### ⚙️ Training Configuration
- Choose base model (Llama 2, etc.)
- Set hyperparameters (epochs, batch size, learning rate)
- Select training method (SFT/RLHF/DPO)
- Toggle demo mode vs. actual training

### 🧪 Inference & Testing
- **Single Prompt**: Generate responses from all 3 models
- **Batch Test**: Evaluate multiple prompts simultaneously
- Compare outputs side-by-side
- Adjust temperature and max tokens

### 📈 Results & Comparison
- **Metrics Tab**: View helpfulness, factuality, safety scores
- **Comparison Tab**: Visual bar charts comparing all methods
- **Analysis Tab**: Key findings and recommendations

## 📁 New Files Created

```
├── streamlit_app.py              # Main Streamlit app (500+ lines)
├── requirements_streamlit.txt    # Streamlit-specific dependencies
├── .streamlit_config.toml        # Streamlit configuration
├── Dockerfile                     # Docker containerization
├── docker-compose.yml            # Docker compose setup
├── STREAMLIT_DEPLOYMENT.md       # Comprehensive deployment guide
├── STREAMLIT_GUIDE.md            # User guide for the dashboard
└── STREAMLIT_README.md           # This file
```

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

**Pros**:
- Free hosting
- Auto-deploys from GitHub
- Works with limited resources
- Custom URL

**Cons**:
- No GPU (training not possible)
- 1GB RAM limit
- Limited storage

**Best for**: Demos, inference, showcasing

```bash
# Just push to GitHub and deploy via streamlit.io
git push origin main
```

### Option 2: Docker + Local

**Pros**:
- Full control
- GPU support
- Unlimited resources

**Cons**:
- Requires local setup
- Not publicly accessible

**Best for**: Development, training

```bash
docker-compose up
# Accessible at http://localhost:8501
```

### Option 3: Docker + Cloud (AWS/GCP)

**Pros**:
- GPU available
- Scalable
- Public URL

**Cons**:
- Costs money
- More complex setup

**Best for**: Production, serious training

```bash
docker build -t llm-training .
# Push to AWS/GCP and deploy
```

See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for detailed instructions.

## 💻 System Requirements

### Minimum (Demo Mode)
- Python 3.10+
- 2GB RAM
- No GPU needed
- Any OS (Windows/Mac/Linux)

### Recommended (Local Training)
- Python 3.10+
- 16GB RAM
- NVIDIA GPU (8GB+ VRAM)
- Linux recommended

### Cloud (Streamlit Cloud)
- Nothing to install!
- Just push to GitHub

## 🎯 Workflow

### Typical User Journey

```
1. Open Dashboard
   ↓
2. Upload Data (📊 Data Curation)
   ↓
3. Filter & Export (Clean data)
   ↓
4. Configure Training (⚙️ Training)
   ↓
5. Start Training (Takes 15-30 min locally)
   ↓
6. Test Models (🧪 Inference)
   ↓
7. Compare Results (📈 Results)
   ↓
8. Download Best Model
```

## 🎨 Features Breakdown

### Data Upload
- JSON format support
- Sample data included
- Preview before processing
- Error handling

### Data Filtering
- Length-based filtering
- Duplicate removal
- Quality heuristics
- Train/test splitting

### Training Interface
- Real-time progress tracking
- Model selection
- Hyperparameter tuning
- Demo vs. actual modes

### Inference Testing
- Single prompt generation
- Batch evaluation
- Side-by-side comparison
- Parameter adjustment

### Results Dashboard
- Multi-metric evaluation
- Visual comparisons
- Statistical analysis
- Downloadable reports

## 🔧 Configuration

### Theme Customization

Edit `.streamlit_config.toml`:

```toml
[theme]
primaryColor = "#0066cc"      # Your brand color
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### App Configuration

Edit `streamlit_app.py` to:
- Change app title/icon
- Modify page content
- Add new pages
- Adjust styling

### Model Configuration

In Training page, adjust:
- Available models
- Default hyperparameters
- Training method options
- Constraints/limits

## 📊 Performance Tips

### Faster Load Times
1. Use demo mode (no actual training)
2. Reduce batch size
3. Use smaller model
4. Cache loaded models

### Better Results
1. Use larger batch size
2. More training epochs
3. Higher quality data
4. GPU acceleration

### Lower VRAM Usage
1. Reduce max_length
2. Reduce batch_size
3. Use LoRA (already done)
4. Use bfloat16 (already done)

## 🔐 Security Notes

### Before Deploying Public

1. **Add Authentication**: If containing sensitive data
   ```python
   import streamlit_authenticator as stauth
   ```

2. **Use Secrets**: Store API keys securely
   ```python
   api_key = st.secrets["huggingface"]["token"]
   ```

3. **Validate Input**: Sanitize user inputs
   ```python
   if len(prompt) > max_length:
       st.error("Prompt too long")
   ```

4. **Rate Limiting**: Prevent abuse
   ```python
   @st.cache_resource(ttl=60)
   def expensive_operation():
       # Cached for 60 seconds
   ```

## 🐛 Common Issues

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements_streamlit.txt
```

### "CUDA out of memory"
```python
# In Training page, reduce batch size:
batch_size = 4  # instead of 8
```

### "Model fails to load"
```bash
huggingface-cli login
# Then enter your HF token
```

### App runs slow
- Use demo mode
- Reduce model size
- Run with `--logger.level=error`

See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for more troubleshooting.

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) | Detailed deployment guide |
| [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) | User guide for the dashboard |
| [README.md](README.md) | Main project documentation |
| [QUICKSTART.md](QUICKSTART.md) | CLI quickstart |

## 🚀 Example Deployments

### Deploy to Streamlit Cloud

```bash
# 1. Push to GitHub
git add streamlit_app.py requirements_streamlit.txt
git commit -m "Add Streamlit dashboard"
git push origin main

# 2. Go to share.streamlit.io
# 3. Click "New app"
# 4. Select your repo and streamlit_app.py
# 5. Deploy!

# Your app will be live at:
# https://yourusername-llm-post-training.streamlit.app
```

### Deploy Locally with Docker

```bash
docker-compose up
# Access at http://localhost:8501
```

### Deploy to AWS

```bash
# See STREAMLIT_DEPLOYMENT.md for full instructions
docker build -t llm-training .
aws ecr create-repository --repository-name llm-training
# ... push to ECR and deploy
```

## 📈 Monetization Ideas

Convert your dashboard into a service:

1. **Free Tier**: Demo mode with limitations
2. **Pro Tier**: Actual model training ($5-10/month)
3. **Enterprise**: Custom models, dedicated GPU
4. **API**: Programmatic access to trained models

## 🎓 What You'll Learn

Building this dashboard teaches:
- Streamlit UI/UX design
- State management in web apps
- File handling (uploads/downloads)
- Data visualization
- Deployment strategies
- Docker containerization
- Cloud deployment (AWS/GCP/Azure)

## 🤝 Contributing

Ideas to extend:

1. **Add Features**
   - Fine-grain metrics
   - Model comparison charts
   - Batch inference API
   - Model merging/export

2. **Improve UX**
   - Dark mode
   - Mobile optimization
   - Progress notifications
   - Real-time training logs

3. **Deployment**
   - Kubernetes support
   - Multi-GPU training
   - Model serving (vLLM)
   - A/B testing framework

## 📞 Support

**Questions?** Check these resources:
- [User Guide](STREAMLIT_GUIDE.md) - How to use the app
- [Deployment Guide](STREAMLIT_DEPLOYMENT.md) - How to deploy
- [GitHub Issues](https://github.com/yourusername/llm-post-training/issues)
- Email: rahulreddy12365@gmail.com

## 📄 License

MIT License - Use freely in your projects!

---

**Ready to go live?** 🚀

Choose your deployment:
- **Cloud (Free)**: `share.streamlit.io`
- **Local**: `streamlit run streamlit_app.py`
- **Docker**: `docker-compose up`

Happy building! 🎉
