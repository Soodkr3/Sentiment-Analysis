# train_sentiment_model.py

import os
import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer  # Updated to TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import string
import logging
import sys
import spacy

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    logging.info("spaCy model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load spaCy model: {e}")
    sys.exit(1)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sentiment_nb_model.joblib")
vectorizer_path = os.path.join(BASE_DIR, "sentiment_vectorizer.joblib")
imdb_dataset_path = os.path.join(BASE_DIR, "data", "IMDB Dataset.csv")  # Update this path if necessary

def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing, removing punctuation,
    tokenizing, removing stopwords, and lemmatizing using spaCy.
    """
    try:
        if not isinstance(text, str):
            return ""
        
        # Lowercase the text
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        
        # Join tokens back to string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    except Exception as e:
        logging.error(f"Error in preprocessing text: {e}")
        return ""

def load_imdb_reviews():
    """
    Loads and preprocesses the IMDb movie reviews dataset.
    Returns a DataFrame with reviews and sentiments.
    """
    try:
        if not os.path.exists(imdb_dataset_path):
            raise FileNotFoundError(f"IMDb dataset not found at {imdb_dataset_path}. Please download it from Kaggle and place it accordingly.")
        
        # Load the IMDb dataset
        df_imdb = pd.read_csv(imdb_dataset_path)
        
        # Ensure the dataset has the expected columns
        if 'review' not in df_imdb.columns or 'sentiment' not in df_imdb.columns:
            raise ValueError("IMDb dataset must contain 'review' and 'sentiment' columns.")
        
        # Map sentiment labels to 'pos' and 'neg' if necessary
        # Assuming the sentiment labels are 'positive' and 'negative'
        df_imdb['sentiment'] = df_imdb['sentiment'].apply(lambda x: 'pos' if str(x).lower() == 'positive' else 'neg')
        
        # Apply preprocessing to the 'review' column
        logging.info("Preprocessing IMDb text data...")
        df_imdb['review'] = df_imdb['review'].apply(preprocess_text)
        
        logging.info(f"Loaded and preprocessed {len(df_imdb)} IMDb reviews.")
        
        return df_imdb
    except Exception as e:
        logging.error(f"Error loading IMDb reviews: {e}")
        raise

def train_or_load_model():
    """
    Trains a Multinomial Naive Bayes model on the IMDb data with hyperparameter tuning,
    or loads the model if it already exists.
    Returns the trained model and vectorizer.
    """
    # Check if cached model and vectorizer exist
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        logging.info("Loading model & vectorizer from cache...")
        model = load(model_path)
        vectorizer = load(vectorizer_path)
    else:
        logging.info("Training new model...")
        
        # Load and preprocess data
        df_imdb = load_imdb_reviews()
        
        # Since we're only using IMDb, no need to combine
        df_combined = df_imdb.copy()
        
        # Display dataset information
        logging.info(f"Combined dataset shape: {df_combined.shape}")
        logging.info(f"Sentiment distribution:\n{df_combined['sentiment'].value_counts()}")
        
        # Initialize TfidfVectorizer with desired parameters
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1,2),
            min_df=5,
            max_df=0.95
        )
        X = vectorizer.fit_transform(df_combined["review"])
        y = df_combined["sentiment"]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize Multinomial Naive Bayes model
        nb = MultinomialNB()
        
        # Define hyperparameter grid
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]  # Smoothing parameter
        }
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=nb,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        logging.info("Starting hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        # Best estimator after GridSearch
        model = grid_search.best_estimator_
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
        
        # Evaluate the best model on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Optimized MultinomialNB Accuracy on Test Set: {accuracy:.4f}")
        logging.info("Classification Report:")
        logging.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save model & vectorizer
        dump(model, model_path)
        dump(vectorizer, vectorizer_path)
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Vectorizer saved to {vectorizer_path}")
    
        # Optional: Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        logging.info(f"Cross-Validation Accuracy Scores: {cv_scores}")
        logging.info(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        
    return model, vectorizer

if __name__ == "__main__":
    model, vectorizer = train_or_load_model()
