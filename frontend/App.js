// src/App.js
import React, { useState } from "react";
import 'bootstrap/dist/css/bootstrap.min.css'; // Bootstrap CSS
import { Spinner, Container, Row, Col, Form, Button, Alert, Card } from "react-bootstrap"; // React Bootstrap Components
import './App.css'; // Custom CSS


function App() {
  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault(); // prevent page refresh

    // Reset previous states
    setPrediction(null);
    setError(null);
    setLoading(true);

    try {
      // Send POST request to Flask
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      // data should look like { sentiment: "pos" } or { sentiment: "neg" }
      setPrediction(data.sentiment);
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to fetch sentiment. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Function to determine the color based on sentiment
  const getSentimentVariant = (sentiment) => {
    if (sentiment === "pos") return "success";
    if (sentiment === "neg") return "danger";
    return "secondary";
  };

  return (
    <Container className="mt-5">
      <Row className="justify-content-md-center">
        <Col md={8} lg={6}>
          <Card>
            <Card.Body>
              <Card.Title className="text-center mb-4">Movie Review Sentiment Analysis</Card.Title>
              <Form onSubmit={handleSubmit}>
                <Form.Group controlId="reviewText">
                  <Form.Label>Enter your movie review:</Form.Label>
                  <Form.Control
                    as="textarea"
                    rows={5}
                    placeholder="Type your review here..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    required
                  />
                </Form.Group>
                <Button variant="primary" type="submit" className="mt-3" disabled={loading}>
                  {loading ? (
                    <>
                      <Spinner
                        as="span"
                        animation="border"
                        size="sm"
                        role="status"
                        aria-hidden="true"
                      />{" "}
                      Predicting...
                    </>
                  ) : (
                    "Predict Sentiment"
                  )}
                </Button>
              </Form>

              {/* Display Error */}
              {error && (
                <Alert variant="danger" className="mt-3">
                  {error}
                </Alert>
              )}

              {/* Display Prediction */}
              {prediction && (
                <Alert variant={getSentimentVariant(prediction)} className="mt-3">
                  <strong>Sentiment:</strong>{" "}
                  {prediction === "pos" ? "Positive 😊" : "Negative 😞"}
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default App;
