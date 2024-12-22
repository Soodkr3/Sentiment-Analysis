from flask import Flask, request, jsonify
from train_and_load_model import train_or_load_model
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Load or train model at startup
model, vectorizer = train_or_load_model()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON in the format: { "text": "I loved this movie!" }
    Returns: { "sentiment": "pos" } or "neg"
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in JSON payload."}), 400
    
    input_text = data["text"]
    text_vector = vectorizer.transform([input_text])
    prediction = model.predict(text_vector)
    
    return jsonify({"sentiment": prediction[0]})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
