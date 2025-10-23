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
# Note: Variable name is CERABUS_API_KEY for backward compatibility, but it's actually for Cerebras API
CERABUS_API_KEY = os.getenv("CERABUS_API_KEY")

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

# -----------------------------
# CERABUS API CALL FUNCTION
# -----------------------------
def call_cerebras_api(prompt, max_tokens=1000):
    """Call Cerebras Inference API for project generation."""
    if not CERABUS_API_KEY:
        st.info("💡 To get AI-generated project ideas, add your CEREBRAS_API_KEY to Streamlit secrets.")
        return None
    
    # Cerebras Inference API endpoint
    url = "https://api.cerebras.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {CERABUS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3.1-8b",  # Fast and efficient model for generation
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates creative and relevant project ideas for students."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract the generated text from Cerebras response
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return None
        
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ AI generation unavailable: Cannot connect to Cerebras API. Please check your internet connection.")
        return None
    except requests.exceptions.HTTPError as e:
        error_msg = f"Status {e.response.status_code}"
        try:
            error_data = e.response.json()
            error_msg += f": {error_data.get('error', {}).get('message', e.response.text[:100])}"
        except:
            error_msg += f": {e.response.text[:100]}"
        st.error(f"🔴 Cerebras API error: {error_msg}")
        return None
    except Exception as e:
        st.error(f"🔴 Unexpected error: {str(e)[:200]}")
        return None

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="AI Project Recommender", page_icon="🤖", layout="wide")

st.title("🤖 AI Project Idea Recommender")
st.write("Get project ideas from a mix of local dataset similarity and AI generation using the Cerebras Inference API.") 

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
use_cerabus = st.checkbox("Use Cerebras AI for fresh project ideas", value=True)

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
        
        # Cerebras AI-generated ideas
        if use_cerabus and CERABUS_API_KEY:
            with st.spinner("✨ Generating fresh AI project ideas with Cerebras..."):
                prompt = f"""Generate 3 unique, creative project ideas for a student interested in: {user_input}

For each project, provide:
1. **Title**: A catchy project name
2. **Description**: 2-3 sentences about what the project does
3. **Domain**: The field (e.g., AI, Web Dev, Data Science, IoT, etc.)
4. **Difficulty**: Beginner, Intermediate, or Advanced
5. **Skills Required**: List of technologies/skills needed

Format each project clearly with markdown headers."""
                
                ai_response = call_cerebras_api(prompt, max_tokens=1000)
                
                if ai_response:
                    st.subheader("Project Ideas")
                    st.markdown(ai_response)
                    
        elif use_cerabus and not CERABUS_API_KEY:
            st.info("💡 To get AI-generated project ideas, add your CEREBRAS_API_KEY to Streamlit secrets.")