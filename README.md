# Netflix Recommendation System

## 1. Overview
This project demonstrates a content-based recommendation system for Netflix titles. It uses two approaches for predicting similar content:

 1. Scikit-learn Implementation: A machine learning-based recommendation system using a Random Forest Classifier.
 2. TensorFlow Implementation: A deep learning-based recommendation system using a Neural Network.

The system recommends similar Netflix titles based on their genres, descriptions, release year, duration, and country of origin.

## 2. Features
- Preprocesses data to extract both textual and metadata features.
- Implements TF-IDF Vectorization to analyze textual data (listed_in and description).
- Uses cosine similarity to measure content similarity.
- Dynamically learns feature weights through machine learning or deep learning.
- Provides recommendations for a given title.

## 3. Requirements
To run this project, you will need:

- Python 3.7 or higher
- Required Python libraries (see requirements.txt)

### Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

## How It Works
1. Dataset: <br>
The dataset should be placed in the data/ folder as netflix_titles.csv. The dataset must include the following columns:

- ```title```: The name of the Netflix title.
- ```listed_in```: The genres/categories of the content.
- ```description```: A brief description of the content.
- ```duration```: Duration of the content (e.g., "90 min" or "2 Seasons").
- ```release_year```: The year the content was released.
- ```country```: The country of origin.
2. Text and Metadata Features: <br>
- Textual Features:
       Combines ```listed_in``` and ```description``` and processes them using TF-IDF Vectorization.
- Metadata Features:
     - ```release_year```: Normalized to a range of [0, 1].
     - ```duration```: Converted to numeric values (e.g., "2 Seasons" → 120 minutes).
     - ```country```: One-hot encoded for regional analysis.
