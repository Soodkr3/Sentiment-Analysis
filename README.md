# Sentiment Analysis Web Application


## Overview
The Sentiment Analysis Web Application is a full-stack machine learning project designed to analyze and classify the sentiment of user-provided text inputs. Whether it's product reviews, social media posts, or any form of textual feedback, this application leverages advanced natural language processing (NLP) techniques to determine the emotional tone behind the words.

<img src="https://github.com/user-attachments/assets/786ab4fc-d87d-4070-9b06-a9de2357c3be" alt="Descriptive Alt Text" width="100" height="100">







## Key Features
+ **User-Friendly Interface**: Built with React.js, the front offers an intuitive and responsive design, allowing users to input text easily and view real-time sentiment results.

+ **Robust Backend API**: Powered by Flask, the backend efficiently handles text processing and model inference and serves predictions via RESTful endpoints.

+ **Machine Learning Model**: Utilizes a pre-trained Naive Bayes classifier from scikit-learn for accurate sentiment analysis, with models serialized using joblib for seamless loading and deployment.

+ **Advanced Text Preprocessing**: Implements spaCy for comprehensive text preprocessing, including tokenization, lemmatization, and stop-word removal, enhancing the model's performance and accuracy.

+ **Cross-Origin Resource Sharing (CORS)**: Configured to allow secure and controlled communication between the front and back end, ensuring smooth data flow and interaction.

+ **JOBLIB**: uses joblib, so there is no need to train the data repetitively.

## Technologies Used
Frontend:

React.js,
 CSS3,
 JavaScript (ES6+)

Backend:

Flask,
 Python 3.x,
 scikit-learn,
 spaCy,
 joblib,


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
python -m spacy download en_core_web_sm
```
Run the Backend Server:

```bash
uvicorn app:app --reload
```
The backend API will be accessible at http://localhost:8000.

3. Setup Frontend
   
Navigate to the Frontend Directory:

```bash
cd ../frontend
```
Install Dependencies:

```bash
npm install
Start the Development Server:
```

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


