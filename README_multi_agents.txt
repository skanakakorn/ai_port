
# AI Portfolio Agent — Multi-Agent Edition

## Run locally
pip install -r requirements.txt
export LLM_PROVIDER=OPENAI
export LLM_MODEL_ID=gpt-4o-mini
export OPENAI_API_KEY=sk-...
# or use TOGETHER/GROQ vars
streamlit run app_multi_agents.py

## Deploy
- Streamlit Community Cloud or Hugging Face Spaces (Streamlit template)
- Cloud Run / Render / Railway: start command
  streamlit run app_multi_agents.py --server.port $PORT --server.address 0.0.0.0
