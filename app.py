import os
import json
import streamlit as st
from dotenv import load_dotenv
from model import load_and_prepare_data, build_vectorizer, recommend_local
import requests
from io import StringIO

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()
CERABUS_API_KEY = os.getenv("CERABUS_API_KEY")

# -----------------------------
# CERABUS API CALL FUNCTION
# -----------------------------
def call_cerabus_api(endpoint, payload=None, method="POST"):
    if not CERABUS_API_KEY:
        st.warning("Cerabus API key not set. Add CERABUS_API_KEY to your .env file to enable AI features.")
        return None
    url = endpoint if endpoint.startswith("http") else f"https://api.cerabus.com/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {CERABUS_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Cerabus API call failed: {e}")
        return None

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="AI Project Recommender", page_icon="🤖", layout="wide")

st.title("🤖 AI Project Idea Recommender (Hybrid: TF-IDF + Cerabus)")
st.write("Get project ideas from a mix of local dataset similarity and AI generation using the Cerabus API.")

# -----------------------------
# Load Dataset + Model
# -----------------------------
DATA_PATH = "project_ideas_dataset_1000.csv"
df = load_and_prepare_data(DATA_PATH)
vectorizer, X = build_vectorizer(df)

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Enter your interests, skills, or goals:", height=120)
use_cerabus = st.checkbox("Use Cerabus AI for fresh project ideas", value=True)

if st.button("🔍 Find Project Ideas"):
    if not user_input.strip():
        st.warning("Please enter your interests or goals first.")
    else:
        # Local recommendations
        with st.spinner("Searching your local dataset..."):
            local_results = recommend_local(user_input, vectorizer, X, df, top_n=5)
        
        st.subheader("🎯 Local Recommendations (TF-IDF Similarity)")
        for _, row in local_results.iterrows():
            st.markdown(f"### 💡 {row['title']}")
            st.markdown(f"**Domain:** {row['domain']} | **Difficulty:** {row['difficulty']} | **Popularity Score:** {row['popularity_score']}")
            st.markdown(f"**Description:** {row['description']}")
            st.markdown(f"**Skills Required:** {row['skills_required']}")