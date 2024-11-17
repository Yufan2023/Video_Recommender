from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
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

# Define and train TensorFlow model
def build_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

tensorflow_model = build_model(X.shape[1])
tensorflow_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Evaluate the TensorFlow model
loss, accuracy = tensorflow_model.evaluate(X_test, y_test, verbose=0)
print(f"TensorFlow Model Accuracy: {accuracy:.2f}")

# Recommendation function
def get_recommendations_tensorflow(title):
    idx = df.index[df['title'] == title].tolist()[0]
    sim_scores = []
    for i in range(len(df)):
        if i != idx:
            text_sim = cosine_sim_text[idx, i]
            metadata_sim = cosine_sim_metadata[idx, i]
            similarity_prob = tensorflow_model.predict([[text_sim, metadata_sim]], verbose=0)[0][0]
            sim_scores.append((i, similarity_prob))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    indices = [i[0] for i in sim_scores[:10]]
    return df['title'].iloc[indices]

example_title = "Kota Factory"
recommendations = get_recommendations_tensorflow(example_title)
print(f"Recommendations (TensorFlow-Based) for '{example_title}':")
for idx, title in enumerate(recommendations, start=1):
    print(f"{idx}. {title}")
