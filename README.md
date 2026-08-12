## Live Demos

| Platform | URL | Notes |
|---|---|---|
| Render | https://langchain-research-copilot.onrender.com/docs | REST API |
| Hugging Face | https://YOUR_HF_USERNAME-langchain-research-copilot.hf.space/docs | ML community |

---

## Deploy Your Own

### Option 1: Render

1. Fork this repository
2. Go to https://render.com → New Web Service
3. Connect your fork
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables from `.env.example`
7. Set `CHROMA_PATH=chroma_db`

### Option 2: Hugging Face Spaces

1. Fork this repository
2. Go to https://huggingface.co → New Space → Docker
3. Push your code to the Space repository
4. Add secrets from `.env.example` in Space settings
5. Set `CHROMA_PATH=/data/chroma_db` as a variable

### Option 3: Docker (local)

```bash
git clone https://github.com/YOUR_USERNAME/langchain-research-copilot
cd langchain-research-copilot
cp .env.example .env
# Fill in your API keys
docker-compose up
```