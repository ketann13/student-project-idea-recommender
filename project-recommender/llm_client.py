"""
Wrapper to call chosen LLM provider (Cerabus only)
"""
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load env for local dev
BASE_DIR = Path(__file__).parent.parent if (Path(__file__).parent / ".env").exists() else Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "project.env")

CERABUS_API_KEY = os.getenv("CERABUS_API_KEY")


def generate_project_ideas(prompt, n=5):
    """
    Generate project ideas using the Cerabus LLM API.
    """
    if not CERABUS_API_KEY:
        raise RuntimeError("CERABUS_API_KEY not set in .env or project.env.")
    url = "https://api.cerabus.com/v1/generate"
    headers = {
        "Authorization": f"Bearer {CERABUS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "n": n
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Expecting a list of ideas in data["ideas"] or similar; adjust as per Cerabus API docs
    return data.get("ideas", data)
