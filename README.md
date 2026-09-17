# 📰 Fake News Detection Using NLP

This project is designed to classify news articles as **fake** or **real** using Natural Language Processing techniques. It uses a Logistic Regression classifier trained on TF-IDF features extracted from news content.

---

## 📁 Dataset

- **Source**: Kaggle (you can use the `Fake.csv` and `True.csv` files or a cleaned merged version)
- **File Used**: `fake_or_real_news_500.csv`  
- The dataset contains labeled news articles categorized as `FAKE` or `REAL`.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 📊 Features

- Text cleaning and preprocessing
- TF-IDF vectorization
- Training/testing split
- Logistic Regression model
- Evaluation using accuracy, confusion matrix, and classification report

---

## 🚀 How to Run the Project

### Deploy online with Render

1. Open [Render](https://render.com/) and sign in with GitHub.
2. Choose **New +** → **Blueprint** and select this repository.
3. Render will read `render.yaml`, install the requirements, and start the Flask website.
4. Open the generated `https://fake-news-detector.onrender.com` URL.

### 1. Clone the Repository
```bash/Terminal
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

###2. Install Required Libraries
```bash/Terminal
pip install -r requirements.txt

###3. Run the Python Script
```bash/Terminal
python fake_news_detector.py


## Output Example

Accuracy: 1.0

Confusion Matrix:
 [[43  0]
 [ 0 57]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        43
           1       1.00      1.00      1.00        57

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100

