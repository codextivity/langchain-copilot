## Live Demos

| Platform | URL | Notes |
| --- | --- | --- |
| Render | https://langchain-copilot.onrender.com/docs | REST API |

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

### Option 2: Docker (local)

```bash
git clone https://github.com/YOUR_USERNAME/langchain-copilot
cd langchain-copilot
cp .env.example .env
# Fill in your API keys
docker-compose up
```