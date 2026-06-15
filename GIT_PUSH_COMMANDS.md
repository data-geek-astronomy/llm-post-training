# Git Push Commands for GitHub

Complete step-by-step commands to push your project to GitHub.

## Step 1: Create Repository on GitHub

1. Go to [github.com/new](https://github.com/new)
2. **Repository name**: `llm-post-training`
3. **Description**: "Production-grade LLM post-training pipeline: SFT, RLHF & DPO"
4. **Visibility**: Public (for maximum visibility/hiring signal)
5. **Add .gitignore**: Python (already in your project)
6. **License**: MIT
7. Click **Create Repository**

Copy the HTTPS URL you see (e.g., `https://github.com/YOUR_USERNAME/llm-post-training.git`)

---

## Step 2: Configure Git (First Time Only)

```bash
# Set your Git identity
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Replace with your actual name and email.

---

## Step 3: Initialize Repository Locally

Navigate to your project folder where all files are:

```bash
cd /Users/aravindkumar/Library/Application\ Support/Claude/local-agent-mode-sessions/b231f330-243d-4d7a-80b6-c3154c476c38/9f2ea253-8c31-40f1-8477-a8f3f6394396/local_47d8671e-f1e3-4401-8960-603bae8ebd90/outputs
```

Then initialize git:

```bash
git init
```

---

## Step 4: Add All Files

```bash
# Add all files
git add .

# Verify files are staged
git status
```

You should see all Python files, markdown docs, Docker configs, etc. in green.

---

## Step 5: Create Initial Commit

```bash
git commit -m "Initial commit: LLM post-training pipeline with SFT, RLHF, DPO and Streamlit dashboard"
```

---

## Step 6: Rename Branch to Main

```bash
git branch -M main
```

---

## Step 7: Add Remote Repository

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/llm-post-training.git
```

**Example**:
```bash
git remote add origin https://github.com/aravind-kumar/llm-post-training.git
```

---

## Step 8: Push to GitHub

```bash
git push -u origin main
```

This will:
- Create `main` branch on GitHub
- Upload all your files
- Set up tracking (so future pushes just need `git push`)

**First time?** You may be prompted to authenticate. Use one of:
- GitHub login credentials
- Personal access token
- SSH key

---

## All Commands in One Block (Copy-Paste Ready)

After creating repo on GitHub and copying the HTTPS URL:

```bash
# Navigate to project folder
cd /Users/aravindkumar/Library/Application\ Support/Claude/local-agent-mode-sessions/b231f330-243d-4d7a-80b6-c3154c476c38/9f2ea253-8c31-40f1-8477-a8f3f6394396/local_47d8671e-f1e3-4401-8960-603bae8ebd90/outputs

# Initialize git
git init

# Add all files
git add .

# Create commit
git commit -m "Initial commit: LLM post-training pipeline with SFT, RLHF, DPO and Streamlit dashboard"

# Set main branch
git branch -M main

# Add remote (REPLACE YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/llm-post-training.git

# Push to GitHub
git push -u origin main
```

---

## Verify It Worked

After pushing, check GitHub:

1. Go to your repo URL: `https://github.com/YOUR_USERNAME/llm-post-training`
2. You should see all your files listed
3. Refresh if needed

---

## Common Issues & Fixes

### "fatal: not a git repository"

Make sure you ran `git init`:
```bash
git init
```

### "Permission denied (publickey)"

Use HTTPS instead of SSH:
```bash
git remote add origin https://github.com/YOUR_USERNAME/llm-post-training.git
```

### "Branch 'main' set up to track remote 'origin/main'"

This is normal. Just means branch is connected. ✅

### "Everything up to date"

Means nothing changed since last push. Good sign. ✅

### Already created repo, want to push?

```bash
git remote add origin https://github.com/YOUR_USERNAME/llm-post-training.git
git branch -M main
git push -u origin main
```

---

## After Initial Push: Future Updates

For future changes:

```bash
# Make changes to files...

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature X or fix bug Y"

# Push to GitHub
git push
```

---

## GitHub Profile Setup (Bonus)

After pushing, make your profile shine:

### Pin Repository to Profile

1. Go to your GitHub profile
2. Click "Customize your pins"
3. Pin `llm-post-training`

### Add Topics (For Discoverability)

1. Go to your repo
2. Click "⚙️ Settings" → "About"
3. Add topics: `llm`, `fine-tuning`, `rlhf`, `dpo`, `lora`

### Write a Compelling README

Your `README.md` is already great! GitHub will display it automatically.

---

## Share Your Project

After pushing, share the link:

**Link to share**: `https://github.com/YOUR_USERNAME/llm-post-training`

**On Twitter**:
```
🚀 Just released my LLM post-training pipeline!

Includes:
• SFT (supervised fine-tuning)
• RLHF (reinforcement learning)
• DPO (direct preference optimization)
• Interactive Streamlit dashboard

All with production-grade code & docs.

GitHub: https://github.com/YOUR_USERNAME/llm-post-training
```

**On LinkedIn**:
```
Excited to share my LLM Post-Training Pipeline project!

This is a complete implementation of modern LLM fine-tuning techniques:
✅ SFT with LoRA for efficiency
✅ RLHF with PPO optimization
✅ DPO as a simpler alternative
✅ Multi-dimensional evaluation framework
✅ Interactive Streamlit dashboard for testing

Check it out on GitHub: [link]
```

---

## What Gets Uploaded

### ✅ Uploaded (Good)
- `src/` - All Python training modules
- `data/` - Sample data files
- `notebooks/` - Educational guides
- `*.md` - All documentation
- `Dockerfile`, `docker-compose.yml` - Deployment configs
- `requirements*.txt` - Dependencies
- `.gitignore` - Git configuration

### ❌ Not Uploaded (Good)
- `results/` - Model checkpoints (too large)
- `__pycache__/` - Python cache
- `.streamlit/secrets.toml` - Private keys
- Large model files (>100MB)

This is correct because `.gitignore` handles it!

---

## Next Steps

1. ✅ **Push code**: Run the commands above
2. ✅ **Add topics**: Make it discoverable
3. ✅ **Pin repo**: Feature on your profile
4. ✅ **Share link**: Tell people about it
5. ✅ **Deploy Streamlit**: Follow STREAMLIT_DEPLOYMENT.md

---

## Quick Reference Card

```bash
# One-time setup
git init
git add .
git commit -m "Initial commit: LLM post-training pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/llm-post-training.git
git push -u origin main

# Future updates
git add .
git commit -m "Your message"
git push
```

---

**You're all set! 🚀**

Need help? Reply to this or check GITHUB_SETUP.md for more details.
