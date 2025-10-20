# Project Recommender

A modular app for recommending project ideas using ML and LLMs (Gemini, Cerabus, etc.).

## Structure

- `app.py`: Streamlit frontend
- `model.py`: ML recommender logic
- `llm_client.py`: LLM API wrapper
- `utils.py`: helpers
- `data/`: project ideas CSV
- `.huggingface/`: (optional) config for HuggingFace Spaces

## Quickstart

1. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the app:
   ```sh
   streamlit run app.py
   ```
