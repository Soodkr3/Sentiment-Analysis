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

### Option A – Docker Compose (recommended)

```bash
git clone https://github.com/soodkr3/Sentiment-Analysis.git
cd Sentiment-Analysis
docker compose up --build
```

- Backend API: http://localhost:8000
- Frontend dev server: http://localhost:3000

> **Note:** The frontend Docker image installs dependencies but the full React build
> requires a `src/` directory restructuring (planned for Stage 2). For full local
> development use Option B below.

---

### Option B – Local development

1. **Clone the repository**

```bash
git clone https://github.com/soodkr3/Sentiment-Analysis.git
cd Sentiment-Analysis
```

2. **Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn enhanced_app:app --reload
```

The backend API will be accessible at http://localhost:8000.

3. **Frontend**

```bash
cd frontend
npm install --legacy-peer-deps
npm start
```

The frontend opens at http://localhost:3000.

### Running tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ --cov=enhanced_app --cov-report=term-missing -v
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Single prediction with confidence scores |
| `POST` | `/predict/batch` | Batch processing with summary statistics |
| `GET`  | `/model/info` | Model architecture and performance metrics |
| `POST` | `/compare` | Compare advanced vs legacy model predictions |
| `GET`  | `/health` | Health check endpoint |

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for enhancements or bug fixes.

## License
This project is licensed under the MIT License.

## Acknowledgements
Special thanks to the creators and maintainers of React, FastAPI, scikit-learn, and Render for providing the tools and platforms that made this project possible.


