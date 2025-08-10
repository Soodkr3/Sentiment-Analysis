# advanced_train_model.py - Enhanced sentiment analysis with multiple models and advanced features

import os
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import string
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "advanced_sentiment_model.joblib")
vectorizer_path = os.path.join(BASE_DIR, "advanced_sentiment_vectorizer.joblib")
model_metrics_path = os.path.join(BASE_DIR, "model_metrics.json")

def create_sample_dataset():
    """
    Creates a sample dataset for training when IMDB dataset is not available.
    Returns a DataFrame with reviews and sentiments.
    """
    sample_data = {
        'review': [
            # Positive samples
            "This movie is absolutely amazing! Great acting and storyline.",
            "I loved every minute of it. Fantastic cinematography.",
            "Excellent film with outstanding performances by all actors.",
            "One of the best movies I've ever seen. Highly recommended!",
            "Brilliant direction and screenplay. A masterpiece!",
            "Wonderful story with incredible visual effects.",
            "Perfect blend of action and emotion. Loved it!",
            "Outstanding movie with great character development.",
            "Exceptional acting and beautiful scenery.",
            "Incredible movie that kept me engaged throughout.",
            "Fantastic plot with amazing twists and turns.",
            "Superb acting and direction. A must-watch!",
            "Beautiful story with excellent execution.",
            "Amazing cinematography and sound design.",
            "Great movie with wonderful performances.",
            "Excellent script and outstanding acting.",
            "Fantastic film with incredible production values.",
            "Wonderful characters and engaging storyline.",
            "Perfect movie for entertainment and inspiration.",
            "Brilliant performances by the entire cast.",
            
            # Negative samples
            "This movie was terrible. Poor acting and boring plot.",
            "Waste of time. Very disappointing and poorly made.",
            "Awful film with bad acting and weak storyline.",
            "One of the worst movies I've ever watched.",
            "Boring and predictable. Not worth watching.",
            "Poor direction and terrible screenplay.",
            "Disappointing movie with bad character development.",
            "Weak plot and unconvincing performances.",
            "Bad acting and poor production quality.",
            "Terrible movie that made no sense.",
            "Boring storyline with poor execution.",
            "Awful direction and weak script.",
            "Disappointing film with bad visual effects.",
            "Poor acting and confusing plot.",
            "Worst movie experience ever.",
            "Terrible screenplay and bad acting.",
            "Boring and poorly directed film.",
            "Awful movie with weak performances.",
            "Disappointing and poorly executed.",
            "Bad film with terrible acting.",
            
            # Additional positive samples
            "Great entertainment value with excellent acting.",
            "Loved the story and the amazing visuals.",
            "Fantastic movie with great emotional depth.",
            "Excellent film that exceeded my expectations.",
            "Wonderful performances and beautiful cinematography.",
            "Amazing story with incredible character arcs.",
            "Perfect movie for all ages. Highly enjoyable!",
            "Outstanding direction and superb acting.",
            "Incredible film with fantastic production quality.",
            "Excellent movie with engaging storyline.",
            
            # Additional negative samples
            "Poor quality film with boring storyline.",
            "Terrible movie that was hard to watch.",
            "Disappointing film with weak character development.",
            "Bad acting and poor direction throughout.",
            "Awful movie with confusing and boring plot.",
            "Worst film I've seen in years.",
            "Poor screenplay and terrible performances.",
            "Boring movie with no redeeming qualities.",
            "Terrible film that was a complete waste of time.",
            "Disappointing movie with bad production values."
        ],
        'sentiment': [
            # Positive labels (30 total)
            'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos',
            'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos',
            # Negative labels (30 total)
            'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg',
            'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg',
            # Additional positive (10)
            'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos',
            # Additional negative (10)
            'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg'
        ]
    }
    return pd.DataFrame(sample_data)

def advanced_preprocess_text(text):
    """
    Advanced text preprocessing without spaCy dependency.
    """
    try:
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except for important ones
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logging.error(f"Error in preprocessing text: {e}")
        return ""

def create_advanced_vectorizer():
    """
    Creates an advanced TF-IDF vectorizer with optimized parameters.
    """
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
        min_df=2,
        max_df=0.95,
        stop_words='english',
        lowercase=True,
        strip_accents='ascii',
        sublinear_tf=True  # Apply sublinear TF scaling
    )

def train_ensemble_model(X_train, y_train):
    """
    Creates and trains an ensemble model combining multiple classifiers.
    """
    # Individual models
    nb_model = MultinomialNB(alpha=0.1)
    lr_model = LogisticRegression(C=10, max_iter=1000, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    
    # Create ensemble using VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('nb', nb_model),
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft'  # Use probability predictions
    )
    
    logging.info("Training ensemble model...")
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation with multiple metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='pos')
    
    logging.info(f"\n{model_name} Evaluation:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }

def train_advanced_model():
    """
    Trains an advanced sentiment analysis model with ensemble methods.
    """
    try:
        # Create or load dataset
        logging.info("Creating sample dataset...")
        df = create_sample_dataset()
        
        # Preprocess text
        logging.info("Preprocessing text data...")
        df['processed_review'] = df['review'].apply(advanced_preprocess_text)
        
        # Display dataset information
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        # Create advanced vectorizer
        vectorizer = create_advanced_vectorizer()
        X = vectorizer.fit_transform(df['processed_review'])
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train ensemble model
        ensemble_model = train_ensemble_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(ensemble_model, X_test, y_test, "Ensemble Model")
        
        # Cross-validation
        logging.info("Performing cross-validation...")
        cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='accuracy')
        logging.info(f"Cross-validation scores: {cv_scores}")
        logging.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model and vectorizer
        dump(ensemble_model, model_path)
        dump(vectorizer, vectorizer_path)
        logging.info(f"Advanced model saved to {model_path}")
        logging.info(f"Advanced vectorizer saved to {vectorizer_path}")
        
        # Save metrics
        import json
        metrics['cv_scores'] = cv_scores.tolist()
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        with open(model_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return ensemble_model, vectorizer
        
    except Exception as e:
        logging.error(f"Error in training advanced model: {e}")
        raise

def test_model_prediction():
    """
    Test the trained model with sample predictions.
    """
    try:
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        
        test_texts = [
            "This movie is absolutely fantastic and amazing!",
            "I hate this boring and terrible film.",
            "The acting was okay but the plot was confusing.",
            "Outstanding performance with brilliant direction.",
            "Worst movie ever made, complete waste of time."
        ]
        
        logging.info("\nTesting model predictions:")
        for text in test_texts:
            processed_text = advanced_preprocess_text(text)
            features = vectorizer.transform([processed_text])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            logging.info(f"Text: {text}")
            logging.info(f"Prediction: {prediction}")
            logging.info(f"Confidence: {confidence:.4f}")
            logging.info(f"Probabilities: neg={probabilities[0]:.4f}, pos={probabilities[1]:.4f}")
            logging.info("-" * 50)
            
    except Exception as e:
        logging.error(f"Error in testing model: {e}")

if __name__ == "__main__":
    try:
        logging.info("Starting advanced sentiment analysis model training...")
        model, vectorizer = train_advanced_model()
        logging.info("Training completed successfully!")
        
        # Test the model
        test_model_prediction()
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)