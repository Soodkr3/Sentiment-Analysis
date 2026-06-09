import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "./App.jsx";

describe("App", () => {
  it("renders the form", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /sentiment analysis/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/movie review/i)).toBeInTheDocument();
  });

  it("shows the prediction returned by the API", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            sentiment: "positive",
            confidence: 0.93,
            probabilities: { positive: 0.93, negative: 0.07 },
            top_features: [{ token: "wonderful", weight: 1.2 }],
          }),
      })
    );

    render(<App />);
    await userEvent.type(screen.getByPlaceholderText(/movie review/i), "wonderful film");
    await userEvent.click(screen.getByRole("button", { name: /analyze sentiment/i }));

    await waitFor(() => {
      expect(screen.getByText(/positive 😊/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/93\.0% confidence/i)).toBeInTheDocument();
    expect(screen.getByText("wonderful")).toBeInTheDocument();

    vi.unstubAllGlobals();
  });
});
