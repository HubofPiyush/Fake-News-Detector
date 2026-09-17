from flask import Flask, jsonify, render_template, request

from fake_news_detector import load_dataset, predict_article, train_detector

app = Flask(__name__)
dataset = load_dataset()
model, vectorizer, evaluation = train_detector(dataset)


@app.get("/")
def dashboard():
    return render_template("index.html")


@app.get("/api/overview")
def overview():
    label_counts = dataset["label"].value_counts().to_dict()
    examples = dataset[["text", "label"]].head(6).to_dict(orient="records")
    return jsonify(
        {
            "total": int(len(dataset)),
            "labels": {label: int(label_counts.get(label, 0)) for label in ["REAL", "FAKE"]},
            "average_length": round(float(dataset["text"].str.len().mean())),
            "vocabulary": int(len(vectorizer.vocabulary_)),
            "metrics": evaluation,
            "examples": examples,
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if len(text) < 20:
        return jsonify({"error": "Paste at least 20 characters of article text."}), 400
    return jsonify(predict_article(text, model, vectorizer))


if __name__ == "__main__":
    app.run(debug=True, port=5000)