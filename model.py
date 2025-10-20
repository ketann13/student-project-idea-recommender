import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1️⃣ Load and Prepare Dataset
# -----------------------------
def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)

    expected_cols = ["project_id", "title", "description", "domain", "skills_required",
                     "difficulty", "goal", "popularity_score", "year"]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df.fillna("", inplace=True)

    df["combined_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["skills_required"] + " " +
        df["goal"]
    )
    return df


# -----------------------------
# 2️⃣ Build TF-IDF model
# -----------------------------
def build_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df["combined_text"])
    return vectorizer, X


# -----------------------------
# 3️⃣ Recommend projects locally
# -----------------------------
def recommend_local(user_input, vectorizer, X, df, top_n=8):
    if X.shape[0] == 0:
        return df.head(0)
    user_vec = vectorizer.transform([user_input])
    sims = cosine_similarity(user_vec, X).flatten()
    indices = sims.argsort()[-top_n:][::-1]
    results = df.iloc[indices].copy()
    results["similarity_score"] = sims[indices]
    return results
