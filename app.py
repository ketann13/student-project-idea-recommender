import os
import json
import streamlit as st
from dotenv import load_dotenv
from model import load_and_prepare_data, build_vectorizer, recommend_local
import requests
from io import StringIO
from pathlib import Path

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()
CERABUS_API_KEY = os.getenv("CERABUS_API_KEY")

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

# -----------------------------
# CERABUS API CALL FUNCTION
# -----------------------------
def call_cerabus_api(endpoint, payload=None, method="POST"):
    if not CERABUS_API_KEY:
        st.info("💡 To get AI-generated project ideas, add your CERABUS_API_KEY to Streamlit secrets.")
        return None
    
    # TODO: Update with correct Cerabus API endpoint
    # Current placeholder: https://api.cerabus.com/v1/generate
    # If using Cerebras or another provider, update the URL below
    url = endpoint if endpoint.startswith("http") else f"https://api.cerabus.com/{endpoint.lstrip('/')}"
    
    headers = {
        "Authorization": f"Bearer {CERABUS_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        st.warning("⚠️ AI generation unavailable: Cannot connect to Cerabus API. Please verify the API endpoint URL or use local recommendations only.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"🔴 Cerabus API error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"🔴 Unexpected error calling Cerabus API: {str(e)[:200]}")
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
# Try multiple possible paths for the CSV file
DATA_PATH = BASE_DIR / "data" / "project_ideas_dataset_1000.csv"
if not DATA_PATH.exists():
    # Fallback to root directory
    DATA_PATH = BASE_DIR / "project_ideas_dataset_1000.csv"
if not DATA_PATH.exists():
    st.error(f"❌ Dataset file not found. Please ensure 'project_ideas_dataset_1000.csv' is in the 'data/' folder or root directory.")
    st.stop()

df = load_and_prepare_data(str(DATA_PATH))
vectorizer, X = build_vectorizer(df)

# Initialize session state for favorites
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔧 Filters & Settings")

# Domain filter
available_domains = sorted(df['domain'].unique())
selected_domains = st.sidebar.multiselect("Filter by Domain", available_domains, default=[])

# Difficulty filter
available_difficulties = sorted(df['difficulty'].unique())
selected_difficulties = st.sidebar.multiselect("Filter by Difficulty", available_difficulties, default=[])

# Number of recommendations
num_recommendations = st.sidebar.slider("Number of recommendations", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Favorites")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        st.sidebar.markdown(f"- {fav}")
else:
    st.sidebar.write("No favorites yet!")

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Enter your interests, skills, or goals:", height=120, placeholder="e.g., machine learning, web development, data science")
use_cerabus = st.checkbox("Use Cerabus AI for fresh project ideas", value=True)

if st.button("🔍 Find Project Ideas"):
    if not user_input.strip():
        st.warning("Please enter your interests or goals first.")
    else:
        # Apply filters
        filtered_df = df.copy()
        if selected_domains:
            filtered_df = filtered_df[filtered_df['domain'].isin(selected_domains)]
        if selected_difficulties:
            filtered_df = filtered_df[filtered_df['difficulty'].isin(selected_difficulties)]
        
        # Rebuild vectorizer for filtered data
        if len(filtered_df) > 0:
            filtered_vectorizer, filtered_X = build_vectorizer(filtered_df)
        else:
            st.warning("No projects match your filters. Try adjusting them.")
            st.stop()
        
        # Local recommendations
        with st.spinner("Searching your local dataset..."):
            local_results = recommend_local(user_input, filtered_vectorizer, filtered_X, filtered_df, top_n=num_recommendations)
        
        st.subheader("🎯 Local Recommendations (TF-IDF Similarity)")
        
        # Export button
        csv_export = local_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_export,
            file_name="project_recommendations.csv",
            mime="text/csv"
        )
        
        for idx, row in local_results.iterrows():
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f"### 💡 {row['title']}")
            with col2:
                if st.button("⭐", key=f"fav_{idx}"):
                    if row['title'] not in st.session_state.favorites:
                        st.session_state.favorites.append(row['title'])
                        st.success(f"Added to favorites!")
                        st.rerun()
            
            st.markdown(f"**Domain:** {row['domain']} | **Difficulty:** {row['difficulty']} | **Popularity Score:** {row['popularity_score']}")
            st.markdown(f"**Description:** {row['description']}")
            st.markdown(f"**Skills Required:** {row['skills_required']}")
            st.markdown("---")
        
        # Cerabus AI-generated ideas
        if use_cerabus and CERABUS_API_KEY:
            with st.spinner("✨ Generating fresh AI project ideas with Cerabus..."):
                prompt = f"Generate 3 unique, creative project ideas for a student interested in: {user_input}. For each project, provide: title, description, domain, difficulty level, and required skills."
                cerabus_response = call_cerabus_api(
                    endpoint="v1/generate",
                    payload={"prompt": prompt, "max_tokens": 1000}
                )
                
                if cerabus_response:
                    st.subheader("🚀 AI-Generated Project Ideas (Cerabus)")
                    
                    # Parse the AI response
                    try:
                        ai_text = cerabus_response.get("text", cerabus_response.get("content", ""))
                        if ai_text:
                            st.markdown(ai_text)
                        else:
                            st.info("AI generated a response but it's in an unexpected format. Raw response:")
                            st.json(cerabus_response)
                    except Exception as e:
                        st.error(f"Error parsing Cerabus response: {e}")
                        st.json(cerabus_response)
        elif use_cerabus and not CERABUS_API_KEY:
            st.info("💡 To get AI-generated project ideas, add your CERABUS_API_KEY to the .env file.")