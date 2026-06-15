# 🚀 Streamlit Dashboard Deployment Guide

Deploy the interactive LLM post-training dashboard on Streamlit Cloud, Docker, or locally.

## Quick Start (Local)

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Run Locally

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🌐 Deploy on Streamlit Cloud (Free & Easy)

### 1. Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"

### 2. Configure App

**Repository**: Select your GitHub repo
**Branch**: `main`
**File path**: `streamlit_app.py`

### 3. Deploy

Click "Deploy" and wait 2-3 minutes.

Your app will be live at: `https://<username>-llm-post-training.streamlit.app`

### Limitations on Streamlit Cloud

⚠️ Important:
- **RAM**: 1 GB (training requires 4-8 GB)
- **CPU**: No GPU (training requires GPU)
- **Storage**: Ephemeral (results don't persist)

**Solution**: Use "Demo Mode" in the app to show pre-computed results.

### Environment Variables (Optional)

Add secrets to `.streamlit/secrets.toml`:

```toml
[huggingface]
token = "hf_xxxxxxxxxxxxx"

[wandb]
api_key = "xxxxxxxxxxxxx"
```

## 🐳 Deploy with Docker

### 1. Build Docker Image

```bash
docker build -t llm-training-dashboard .
```

### 2. Run Container Locally

```bash
docker run -p 8501:8501 llm-training-dashboard
```

Access at `http://localhost:8501`

### 3. With GPU Support

```bash
docker run --gpus all -p 8501:8501 llm-training-dashboard
```

### 4. Using Docker Compose

```bash
docker-compose up
```

Automatically exposes port 8501 and creates volumes for data/results.

## ☁️ Deploy on Cloud Platforms

### AWS (ECS)

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name llm-training

# 2. Build and push
docker build -t llm-training-dashboard .
docker tag llm-training-dashboard:latest <account>.dkr.ecr.<region>.amazonaws.com/llm-training:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/llm-training:latest

# 3. Create ECS task definition and service
# (See AWS console for detailed steps)
```

### Google Cloud Run

```bash
# 1. Set up gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/llm-training

# 3. Deploy
gcloud run deploy llm-training \
  --image gcr.io/YOUR_PROJECT_ID/llm-training \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Heroku

```bash
# 1. Login to Heroku
heroku login

# 2. Create app
heroku create llm-training-app

# 3. Push code
git push heroku main

# 4. View logs
heroku logs --tail
```

Note: Free tier Heroku has been deprecated. Consider using Railway, Render, or similar.

### Azure Container Instances

```bash
# 1. Create container registry
az acr create --resource-group mygroup --name myregistry --sku Basic

# 2. Build and push
az acr build --registry myregistry --image llm-training .

# 3. Deploy
az container create \
  --resource-group mygroup \
  --name llm-training \
  --image myregistry.azurecr.io/llm-training:latest \
  --ports 8501
```

## 📊 Full Training (GPU Required)

For actual model training (not just demo), you need:
- **GPU**: NVIDIA A100, H100, or RTX 4090
- **RAM**: 16GB+ system memory
- **VRAM**: 8-16GB GPU memory

### Local GPU Setup

```bash
# 1. Install CUDA & cuDNN
# See: https://pytorch.org/get-started/locally/

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Run training
python src/training/sft.py \
  --model_name meta-llama/Llama-2-7b \
  --data_path data/processed/train.json \
  --output_dir results/sft
```

### Cloud GPU Instance (AWS)

```bash
# 1. Launch EC2 GPU instance
# AMI: Deep Learning AMI (Ubuntu 22.04) with CUDA pre-installed
# Instance: g4dn.xlarge or g4dn.2xlarge

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Clone repo and train
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -r requirements.txt
python src/training/sft.py ...
```

## 🔄 CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Push to Streamlit Cloud
        run: |
          # Streamlit Cloud auto-redeploys on push
          echo "Streamlit Cloud auto-deploying from main branch"
```

## 📋 Deployment Checklist

### Before Deployment

- [ ] Test app locally: `streamlit run streamlit_app.py`
- [ ] Check all imports work
- [ ] Test file uploads
- [ ] Verify data processing
- [ ] Test inference (demo mode)
- [ ] Check responsive design (mobile)

### Streamlit Cloud Specific

- [ ] Create `.streamlit/secrets.toml` with API keys
- [ ] Set up GitHub repo with proper name
- [ ] Ensure `requirements_streamlit.txt` exists
- [ ] Add `streamlit_app.py` to main directory
- [ ] Push changes to GitHub
- [ ] Grant Streamlit permission to access repo

### Docker Specific

- [ ] Dockerfile builds without errors
- [ ] Image runs locally: `docker run ...`
- [ ] Port 8501 is accessible
- [ ] Health check works
- [ ] Volumes mounted correctly (if needed)

### Cloud Platform Specific

- [ ] Set environment variables
- [ ] Configure resource limits
- [ ] Set up logging
- [ ] Configure auto-scaling (if needed)
- [ ] Set up monitoring/alerts
- [ ] Configure backup/persistence

## 🚨 Troubleshooting

### "ModuleNotFoundError"

```bash
pip install -r requirements_streamlit.txt
```

### "CUDA out of memory" (GPU)

In Streamlit Cloud: Not possible (no GPU). Use demo mode.

Locally: Reduce batch size in training config.

### "Model not found" (Hugging Face)

```bash
huggingface-cli login
# Then enter your HF token
```

### "Port 8501 already in use"

```bash
streamlit run streamlit_app.py --server.port 8502
```

### Slow performance

**Streamlit Cloud**: Use lighter models or demo mode
**Docker**: Increase memory allocation: `docker run -m 4g ...`
**GPU**: Enable GPU with `--gpus all`

## 📊 Monitoring & Scaling

### Streamlit Cloud

View app stats in Streamlit Cloud dashboard:
- User sessions
- Execution time
- Memory usage
- CPU usage

### Docker / Self-Hosted

Add monitoring:

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## 💰 Cost Comparison

| Platform | Cost | GPU | Best For |
|----------|------|-----|----------|
| Streamlit Cloud | Free | ❌ | Demo/inference only |
| Docker + Local | Hardware | ✅ | Full training, dev |
| AWS EC2 (g4dn) | $0.53/hr | ✅ | Production training |
| Google Cloud Run | Pay-per-use | ❌ | Inference at scale |
| Heroku (alternative) | Varies | ❌ | Simple deployment |

## 🎯 Best Practices

1. **Version Your Dependencies**
   ```bash
   pip freeze > requirements_streamlit.txt
   ```

2. **Use Environment Variables**
   ```python
   import os
   api_key = os.getenv('HF_TOKEN')
   ```

3. **Add Error Handling**
   ```python
   try:
       # code
   except Exception as e:
       st.error(f"Error: {e}")
   ```

4. **Cache Expensive Operations**
   ```python
   @st.cache_resource
   def load_model():
       # Load once, reuse across sessions
       return model
   ```

5. **Monitor Resources**
   ```python
   import psutil
   st.write(f"Memory: {psutil.virtual_memory().percent}%")
   ```

## 🚀 Next Steps

1. **Test locally** - Run `streamlit run streamlit_app.py`
2. **Deploy to Streamlit Cloud** - Free & easy
3. **Set up custom domain** - Optional, via Streamlit Pro
4. **Add authentication** - If needed for security
5. **Set up analytics** - Track user engagement

## 📚 References

- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Cloud](https://share.streamlit.io)
- [Docker Docs](https://docs.docker.com)
- [Hugging Face Model Hub](https://huggingface.co/models)

---

Need help? Email rahulreddy12365@gmail.com or open a GitHub issue.
