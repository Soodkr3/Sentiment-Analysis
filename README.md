# Sentiment Analysis Web Application


## Overview
The Sentiment Analysis Web Application is a full-stack machine learning project designed to analyze and classify the sentiment of user-provided text inputs. Whether it's product reviews, social media posts, or any form of textual feedback, this application leverages advanced natural language processing (NLP) techniques to determine the emotional tone behind the words.


<img src="https://github.com/user-attachments/assets/786ab4fc-d87d-4070-9b06-a9de2357c3be" alt="Descriptive Alt Text" width="400" height="300" >

<img src="https://github.com/user-attachments/assets/bca07a64-4be1-4e55-8979-b34350bb6f9e" alt="Descriptive Alt Text" width="400" height="300" >






## Key Features
+ **Advanced Machine Learning**: Enhanced ensemble model combining Naive Bayes, Logistic Regression, and Random Forest for superior accuracy (96.7% cross-validation score)

+ **Confidence Scoring**: Every prediction includes confidence levels and probability distributions for both positive and negative sentiments

+ **Batch Processing**: Analyze multiple texts simultaneously with comprehensive summary statistics including sentiment distribution and average confidence

+ **User-Friendly Interface**: Built with React.js, the front offers an intuitive and responsive design, allowing users to input text easily and view real-time sentiment results with enhanced metrics

+ **Robust Backend API**: Powered by FastAPI, the backend efficiently handles text processing and model inference with multiple endpoints for different use cases

+ **Advanced Text Preprocessing**: Implements TF-IDF vectorization with n-grams (1-3) for comprehensive feature extraction, enhancing the model's performance and accuracy

+ **Model Comparison**: Compare predictions between advanced ensemble model and legacy model to see accuracy improvements

+ **Cross-Origin Resource Sharing (CORS)**: Configured to allow secure and controlled communication between the front and back end, ensuring smooth data flow and interaction

+ **Backward Compatibility**: Enhanced models while maintaining full compatibility with existing frontend applications

## Technologies Used
### Frontend:

+ React.js
+ CSS3
+ JavaScript (ES6+)

### Backend:

+ FastAPI (upgraded from Flask)
+ Python 3.x
+ scikit-learn (ensemble models: Naive Bayes + Logistic Regression + Random Forest)
+ Advanced TF-IDF vectorization with n-grams
+ joblib
+ Cross-validation and model comparison capabilities


## Getting Started
To set up and run the project locally, follow these steps:

1. Clone the Repository

```bash
git clone https://github.com/soodkr3/Sentiment-Analysis.git
cd Sentiment-Analysis
```

2. Setup Backend

Navigate to the Backend Directory:

```bash
cd backend
```
Create and Activate a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Note: spaCy is no longer required for the advanced model
```
Run the Backend Server:

```bash
# For enhanced API with all advanced features
uvicorn enhanced_app:app --reload

# OR for backward compatible API 
uvicorn app:app --reload
```
The backend API will be accessible at http://localhost:8000.

### New API Endpoints:
+ `POST /predict` - Single prediction with confidence scores
+ `POST /predict/batch` - Batch processing with summary statistics  
+ `GET /model/info` - Model architecture and performance metrics
+ `POST /compare` - Compare advanced vs legacy model predictions
+ `GET /health` - Health check endpoint

3. Setup Frontend
   
Navigate to the Frontend Directory:

```bash
cd ../frontend
```
Install Dependencies:

```bash
npm install
```

Start the Development Server:

```bash
npm start
```
The frontend application will open in your default browser at http://localhost:3000.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for enhancements or bug fixes.

## License
This project is licensed under the MIT License.

## Acknowledgements
Special thanks to the creators and maintainers of React, FastAPI, scikit-learn, spaCy, and Render for providing the tools and platforms that made this project possible.


