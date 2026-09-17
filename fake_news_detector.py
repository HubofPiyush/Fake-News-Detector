from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).with_name("fake_or_real_news_500.csv")


def load_dataset(path=DATASET_PATH):
	"""Load and normalize the small labelled news dataset."""
	frame = pd.read_csv(path)
	frame = frame.dropna(subset=["text", "label"]).copy()
	frame["text"] = frame["text"].astype(str).str.strip()
	frame["label"] = frame["label"].astype(str).str.upper().str.strip()
	return frame[frame["label"].isin(["FAKE", "REAL"])]


def train_detector(frame=None):
	"""Train the detector and return the model plus dashboard-ready metrics."""
	frame = load_dataset() if frame is None else frame
	features_train, features_test, labels_train, labels_test = train_test_split(
		frame["text"],
		frame["label"],
		test_size=0.2,
		random_state=42,
		stratify=frame["label"],
	)
	vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
	train_vectors = vectorizer.fit_transform(features_train)
	test_vectors = vectorizer.transform(features_test)
	model = LogisticRegression(max_iter=1000)
	model.fit(train_vectors, labels_train)
	predictions = model.predict(test_vectors)
	matrix = confusion_matrix(labels_test, predictions, labels=["FAKE", "REAL"])
	report = classification_report(
		labels_test, predictions, labels=["FAKE", "REAL"], output_dict=True, zero_division=0
	)
	metrics = {
		"accuracy": round(float(accuracy_score(labels_test, predictions)) * 100, 1),
		"tested": int(len(labels_test)),
		"confusion_matrix": matrix.tolist(),
		"report": {
			label: {
				"precision": round(float(report[label]["precision"]) * 100, 1),
				"recall": round(float(report[label]["recall"]) * 100, 1),
				"f1": round(float(report[label]["f1-score"]) * 100, 1),
				"support": int(report[label]["support"]),
			}
			for label in ["FAKE", "REAL"]
		},
	}
	return model, vectorizer, metrics


def predict_article(text, model, vectorizer):
	"""Classify one article and return its label and confidence."""
	probabilities = model.predict_proba(vectorizer.transform([text]))[0]
	class_index = int(probabilities.argmax())
	return {
		"label": str(model.classes_[class_index]),
		"confidence": round(float(probabilities[class_index]) * 100, 1),
		"probabilities": {
			str(label): round(float(probability) * 100, 1)
			for label, probability in zip(model.classes_, probabilities)
		},
	}


if __name__ == "__main__":
	dataset = load_dataset()
	_, _, evaluation = train_detector(dataset)
	print("Data preview:")
	print(dataset.head())
	print("\nLabel distribution:")
	print(dataset["label"].value_counts())
	print(f"\nAccuracy: {evaluation['accuracy']}%")
	print("\nConfusion Matrix (FAKE, REAL):")
	print(evaluation["confusion_matrix"])
	print("\nClassification Report:")
	for label, scores in evaluation["report"].items():
		print(label, scores)
