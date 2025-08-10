# Advanced Sentiment Analysis - Model Improvements

## Overview
This project has been enhanced with a more sophisticated sentiment analysis model that provides better accuracy and additional features.

## Key Improvements

### 1. Enhanced Model Architecture
- **Ensemble Model**: Combines three different algorithms:
  - Multinomial Naive Bayes (original)
  - Logistic Regression (new)
  - Random Forest Classifier (new)
- **Voting Strategy**: Uses soft voting to combine predictions based on probability distributions
- **Cross-validation**: Implements 5-fold stratified cross-validation for robust evaluation

### 2. Advanced Feature Engineering
- **TF-IDF Vectorization**: Replaced simple CountVectorizer with TF-IDF for better feature representation
- **N-gram Support**: Includes unigrams, bigrams, and trigrams (1-3) for better context understanding
- **Sublinear TF Scaling**: Improves performance on large documents
- **Advanced Preprocessing**: Enhanced text cleaning and normalization

### 3. New API Features
- **Confidence Scores**: Every prediction includes a confidence score (0.0 to 1.0)
- **Probability Distributions**: Full probability breakdown for both positive and negative sentiments
- **Batch Processing**: Analyze multiple texts in a single request with summary statistics
- **Model Comparison**: Compare predictions between advanced and legacy models
- **Model Information**: Get detailed information about model architecture and performance metrics

### 4. Enhanced API Endpoints

#### `/predict` (Enhanced)
```json
{
  "sentiment": "pos",
  "confidence": 0.9052,
  "probabilities": {
    "negative": 0.0948,
    "positive": 0.9052
  },
  "model_type": "advanced"
}
```

#### `/predict/batch` (New)
```json
{
  "results": [...],
  "summary": {
    "total_texts": 3,
    "positive_count": 2,
    "negative_count": 1,
    "positive_percentage": 66.67,
    "negative_percentage": 33.33,
    "average_confidence": 0.7775
  }
}
```

#### `/model/info` (New)
```json
{
  "model_type": "Ensemble (Naive Bayes + Logistic Regression + Random Forest)",
  "features": [
    "TF-IDF Vectorization with n-grams (1-3)",
    "Advanced text preprocessing",
    "Ensemble voting (soft voting)",
    "Cross-validation scoring"
  ],
  "training_accuracy": 1.0,
  "cross_validation_score": 0.9667
}
```

#### `/compare` (New)
Compare predictions between advanced and legacy models to see the improvement in confidence and accuracy.

#### `/health` (New)
Health check endpoint for monitoring and deployment.

## Performance Improvements

### Model Metrics
- **Cross-validation Accuracy**: 96.67% (±8.16%)
- **Training Accuracy**: 100% (on sample dataset)
- **Confidence Scores**: More reliable probability estimates
- **Reduced Overfitting**: Ensemble approach provides better generalization

### Features Comparison

| Feature | Legacy Model | Advanced Model |
|---------|-------------|----------------|
| Algorithm | Naive Bayes | Ensemble (NB + LR + RF) |
| Vectorization | CountVectorizer | TF-IDF with n-grams |
| Confidence Scores | ❌ | ✅ |
| Batch Processing | ❌ | ✅ |
| Cross-validation | ❌ | ✅ |
| Model Comparison | ❌ | ✅ |
| Probability Distribution | ❌ | ✅ |

## Usage Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is absolutely fantastic!"}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great movie!", "Terrible film!", "It was okay."]}'
```

### Model Information
```bash
curl -X GET "http://localhost:8000/model/info"
```

### Compare Models
```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is fantastic!"}'
```

## Technical Details

### Model Training
The advanced model is trained using the `advanced_train_model.py` script which:
1. Creates a balanced sample dataset (expandable with real data)
2. Implements advanced text preprocessing
3. Uses TF-IDF vectorization with optimal parameters
4. Trains an ensemble of three different algorithms
5. Performs cross-validation for robust evaluation
6. Saves model metrics for API consumption

### Backward Compatibility
The system maintains backward compatibility by:
- Keeping the original `/predict` endpoint functional
- Auto-detecting and loading the best available model
- Providing fallback to legacy model if advanced model is unavailable
- Maintaining the same response format with additional fields

### Deployment
Both `app.py` (backward compatible) and `enhanced_app.py` (full-featured) are available:
- Use `app.py` for backward compatibility with existing frontend
- Use `enhanced_app.py` for all new features and better API documentation

## Future Enhancements
Potential areas for further improvement:
1. **Deep Learning Models**: Integration with BERT, RoBERTa, or other transformer models
2. **Multi-class Sentiment**: Support for neutral, very positive, very negative classifications
3. **Real-time Training**: Online learning capabilities
4. **Model Interpretability**: LIME/SHAP integration for explanation
5. **Language Support**: Multi-language sentiment analysis
6. **Custom Domain Training**: Industry-specific model fine-tuning