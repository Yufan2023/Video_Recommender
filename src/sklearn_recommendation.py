import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_and_preprocess_data

# Load and preprocess data
file_path = "../data/netflix_titles.csv"
df, tfidf_matrix, metadata_matrix = load_and_preprocess_data(file_path)

# Generate pairwise data
from sklearn.metrics.pairwise import cosine_similarity
import itertools

cosine_sim_text = cosine_similarity(tfidf_matrix)
cosine_sim_metadata = cosine_similarity(metadata_matrix)

pairs = list(itertools.combinations(range(len(df)), 2))
sample_pairs = np.random.choice(len(pairs), size=10000, replace=False)
sampled_pairs = [pairs[i] for i in sample_pairs]

X = []
y = []

for idx1, idx2 in sampled_pairs:
    text_sim = cosine_sim_text[idx1, idx2]
    metadata_sim = cosine_sim_metadata[idx1, idx2]
    label = 1 if df.iloc[idx1]['listed_in'] == df.iloc[idx2]['listed_in'] else 0
    X.append([text_sim, metadata_sim])
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy of the model: {accuracy:.2f}")

# Recommendation function
def get_recommendations_ml(title):
    idx = df.index[df['title'] == title].tolist()[0]
    sim_scores = []
    for i in range(len(df)):
        if i != idx:
            text_sim = cosine_sim_text[idx, i]
            metadata_sim = cosine_sim_metadata[idx, i]
            sim_scores.append((i, model.predict_proba([[text_sim, metadata_sim]])[0][1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    indices = [i[0] for i in sim_scores[:10]]
    return df['title'].iloc[indices]

example_title = "Kota Factory"  # Replace with a valid title
recommendations = get_recommendations_ml(example_title)
print(f"Recommendations for '{example_title}':\n{recommendations}")
