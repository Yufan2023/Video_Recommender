import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Combine 'listed_in' and 'description' for textual similarity
    df['content'] = df['listed_in'].fillna('') + " " + df['description'].fillna('')

    # Initialize TF-IDF Vectorizer for text features
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df['content'])

    # Convert 'duration' to numeric
    def process_duration(duration):
        if pd.isnull(duration):
            return 0
        if "Season" in duration:
            return int(duration.split()[0]) * 60
        if "min" in duration:
            return int(duration.split()[0])
        return 0

    df['duration_numeric'] = df['duration'].apply(process_duration)

    # Normalize duration, release_year, and country metadata
    scaler = MinMaxScaler()
    metadata_features = ['duration_numeric', 'release_year']
    df[metadata_features] = scaler.fit_transform(df[metadata_features])

    # Convert country into dummy variables (one-hot encoding)
    country_dummies = pd.get_dummies(df['country'], prefix='country', dummy_na=True)

    # Combine metadata features
    metadata_matrix = np.hstack([
        tfidf_matrix.toarray(),
        df[metadata_features].values,
        country_dummies.values
    ])

    return df, tfidf_matrix, metadata_matrix
