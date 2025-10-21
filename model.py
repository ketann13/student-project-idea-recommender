import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Text Preprocessing
# -----------------------------
def clean_text(text):
    """Clean and normalize text for better matching."""
    if not isinstance(text, str):
        return ""
    if text.strip() == "":
        return ""
    # Convert to lowercase
    text = text.lower()
    # Keep alphanumeric, spaces, and hyphens (important for tech terms like "next-gen", "AI-powered")
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# -----------------------------
# 1️⃣ Load and Prepare Dataset
# -----------------------------
def load_and_prepare_data(csv_file):
    """Load CSV and prepare data with text cleaning."""
    df = pd.read_csv(csv_file)

    expected_cols = ["project_id", "title", "description", "domain", "skills_required",
                     "difficulty", "goal", "popularity_score", "year"]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df.fillna("", inplace=True)

    # Create combined text with weighted importance (title gets more weight)
    df["combined_text"] = (
        df["title"].astype(str) + " " + 
        df["title"].astype(str) + " " +  # Title appears twice for more weight
        df["description"].astype(str) + " " +
        df["skills_required"].astype(str) + " " +
        df["goal"].astype(str) + " " +
        df["domain"].astype(str)
    )
    
    # Ensure no empty strings in combined_text
    df["combined_text"] = df["combined_text"].str.strip()
    df["combined_text"] = df["combined_text"].replace("", "unknown project data science")
    
    return df


# -----------------------------
# 2️⃣ Build TF-IDF model
# -----------------------------
def build_vectorizer(df):
    """Build TF-IDF vectorizer with improved parameters."""
    # Check if combined_text has content
    if df["combined_text"].str.len().sum() == 0:
        raise ValueError("Combined text column is empty. Check your CSV data.")
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=1,
        max_df=1.0,  # Changed from 0.8 to allow more common terms
        token_pattern=r'\b\w+\b',  # More lenient token pattern
        lowercase=True
    )
    X = vectorizer.fit_transform(df["combined_text"])
    
    return vectorizer, X


# -----------------------------
# 3️⃣ Recommend projects locally
# -----------------------------
def recommend_local(user_input, vectorizer, X, df, top_n=8):
    """Find most similar projects using cosine similarity."""
    if X.shape[0] == 0:
        return df.head(0)
    
    # Clean user input
    cleaned_input = clean_text(user_input)
    user_vec = vectorizer.transform([cleaned_input])
    
    # Calculate cosine similarity
    sims = cosine_similarity(user_vec, X).flatten()
    
    # Get top N indices
    indices = sims.argsort()[-top_n:][::-1]
    
    # Filter out very low similarity scores (threshold = 0.01)
    valid_indices = [i for i in indices if sims[i] > 0.01]
    
    if not valid_indices:
        # If no good matches, return top N anyway
        valid_indices = indices
    
    results = df.iloc[valid_indices].copy()
    results["similarity_score"] = sims[valid_indices]
    
    # Round similarity score for display
    results["similarity_score"] = results["similarity_score"].round(3)
    
    return results


# -----------------------------
# 4️⃣ Find similar projects
# -----------------------------
def find_similar_projects(project_id, df, X, top_n=5):
    """Find projects similar to a given project."""
    try:
        idx = df[df['project_id'] == project_id].index[0]
        project_vec = X[idx]
        sims = cosine_similarity(project_vec, X).flatten()
        
        # Exclude the project itself
        sims[idx] = -1
        
        indices = sims.argsort()[-top_n:][::-1]
        similar_projects = df.iloc[indices].copy()
        similar_projects["similarity_score"] = sims[indices].round(3)
        
        return similar_projects
    except IndexError:
        return pd.DataFrame()
