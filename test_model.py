from model import load_and_prepare_data, build_vectorizer

# Test loading and vectorization
df = load_and_prepare_data('data/project_ideas_dataset_1000.csv')
print(f'Loaded {len(df)} rows')
print(f'Sample combined_text: {df["combined_text"].iloc[0][:100]}...')
print(f'Empty texts: {(df["combined_text"].str.len() == 0).sum()}')

vectorizer, X = build_vectorizer(df)
print(f'Vectorizer shape: {X.shape}')
print('✅ Model loading and vectorization successful!')
