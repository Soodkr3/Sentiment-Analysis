import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const EXAMPLES = [
  "An absolute triumph — the performances were stunning and the script never missed a beat.",
  "Two hours of my life I will never get back. Dull, predictable, and badly acted.",
  "The cinematography was gorgeous, but the plot was not good and the pacing dragged.",
];

function ConfidenceBar({ probabilities }) {
  const positive = (probabilities.positive ?? 0) * 100;
  return (
    <div className="confidence-bar" aria-label="probability split">
      <div className="confidence-bar__negative" style={{ width: `${100 - positive}%` }}>
        {(100 - positive).toFixed(1)}% negative
      </div>
      <div className="confidence-bar__positive" style={{ width: `${positive}%` }}>
        {positive.toFixed(1)}% positive
      </div>
    </div>
  );
}

function TokenChips({ features }) {
  if (!features?.length) return null;
  return (
    <div className="tokens">
      <h3>Why the model decided this</h3>
      <p className="tokens__hint">
        The classifier is linear, so these are the exact words (and word pairs) that
        pushed this prediction — green towards positive, red towards negative.
      </p>
      <ul>
        {features.map(({ token, weight }) => (
          <li
            key={token}
            className={weight >= 0 ? "token token--positive" : "token token--negative"}
          >
            {token}
            <span className="token__weight">
              {weight >= 0 ? "+" : ""}
              {weight.toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyze = async (event) => {
    event.preventDefault();
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      console.error(err);
      setError("Could not reach the API. Is the backend running on " + API_URL + "?");
    } finally {
      setLoading(false);
    }
  };

  const isPositive = result?.sentiment === "positive";

  return (
    <main className="app">
      <header>
        <h1>Sentiment Analysis</h1>
        <p className="subtitle">
          TF-IDF + logistic regression trained on 50k IMDB reviews, with exact
          per-token explanations.
        </p>
      </header>

      <form onSubmit={analyze}>
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Paste a movie review here..."
          rows={6}
          required
        />
        <div className="actions">
          <button type="submit" disabled={loading || !text.trim()}>
            {loading ? "Analyzing..." : "Analyze sentiment"}
          </button>
          <div className="examples">
            {EXAMPLES.map((example, index) => (
              <button
                key={index}
                type="button"
                className="example"
                onClick={() => setText(example)}
              >
                Example {index + 1}
              </button>
            ))}
          </div>
        </div>
      </form>

      {error && <div className="error" role="alert">{error}</div>}

      {result && (
        <section className="result">
          <div className="result__headline">
            <span className={`badge ${isPositive ? "badge--positive" : "badge--negative"}`}>
              {isPositive ? "Positive 😊" : "Negative 😞"}
            </span>
            <span className="result__confidence">
              {(result.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>
          <ConfidenceBar probabilities={result.probabilities} />
          <TokenChips features={result.top_features} />
        </section>
      )}
    </main>
  );
}
