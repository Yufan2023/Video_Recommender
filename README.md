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

## 4. How It Works
1. Dataset: <br>
The dataset should be placed in the data/ folder as netflix_titles.csv. The dataset must include the following columns:

- ```title```: The name of the Netflix title.
- ```listed_in```: The genres/categories of the content.
- ```description```: A brief description of the content.
- ```duration```: Duration of the content (e.g., "90 min" or "2 Seasons").
- ```release_year```: The year the content was released.
- ```country```: The country of origin.
2. Text and Metadata Features: <br>
- Textual Features: <br>
       Combines ```listed_in``` and ```description``` and processes them using TF-IDF Vectorization.
- Metadata Features:
     - ```release_year```: Normalized to a range of [0, 1].
     - ```duration```: Converted to numeric values (e.g., "2 Seasons" → 120 minutes).
     - ```country```: One-hot encoded for regional analysis.
 
## 5. Models
1. Scikit-learn (Random Forest Classifier)
   - Uses cosine similarity from textual and metadata features as inputs.
   - Predicts whether pairs of titles are similar using a Random Forest Classifier.
   - Generates recommendations based on learned weights.
2. TensorFlow (Neural Network)
   - Uses a deep neural network to dynamically predict similarity probabilities.
   - Incorporates textual and metadata similarities as input features.
  
## 6. Usage
### 1. Run the Scikit-learn Implementation
Execute the following command: <br>
```bash
python src/sklearn_recommendation.py
```
### 2. Run the TensorFlow Implementation
Execute the following command: <br>
```bash
python src/tensorflow_recommendation.py
```

## 7. Customization
1. Change the Input Title: <br>
Modify the ```example_title``` variable in ```sklearn_recommendation.py``` or ```tensorflow_recommendation.py``` to get recommendations for a different title:
   ```bash
   example_title = "Squid Game"  # Replace with a valid title
   ```
2. Adjust Model Parameters:<br>
- In Scikit-learn, tweak the Random Forest parameters (e.g., n_estimators, max_depth).
- In TensorFlow, modify the neural network architecture (e.g., number of layers, neurons, or epochs).


