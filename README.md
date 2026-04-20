# Predicting User Sentiment

A deep learning system that predicts the **sentiment of tweets** from specific public figures. Instead of using one global model, it trains a **separate LSTM model per user**, allowing the system to learn each user's unique tone and vocabulary patterns.

---

## What the Project Does

Given a tweet and a username, the system classifies the tweet sentiment as:
- **Positive**
- **Neutral**
- **Negative**

The project covers the full pipeline — from raw tweet CSVs all the way to trained, saved models ready for inference.

---

## Data Sources

The project starts with **12 raw CSV files**, each containing tweets from one public figure:

| # | User |
|---|------|
| 1 | Elon Musk |
| 2 | Imran Khan (ImranKhanPTI) |
| 3 | Cristiano Ronaldo |
| 4 | Kanye West |
| 5 | Lionel Messi |
| 6 | Meghan Markle |
| 7 | Sanna Marin |
| 8 | Joe Biden |
| 9 | Jair Bolsonaro |
| 10 | Ben Hamner |
| 11 | Drill Tweets |
| 12 | CR (alternate Ronaldo source) |

After filtering and curation, the final training dataset contains **2,827 tweets** from 3 users:

| User | Tweets |
|---|---|
| Elon Musk | 2,527 |
| ImranKhanPTI | 267 |
| Cristiano Ronaldo | 33 |

---

## Data Preprocessing

### Steps

**1. Merge**
All 12 individual CSV files are merged into a single dataset, with a `user_name` column added to each row.

**2. Standardize Datetime**
Timestamps are converted to a consistent `YYYY-MM-DD HH:MM:SS` format.

**3. Column Filtering**
Only three columns are kept: `user_name`, `time_date`, `text`.

**4. Sentiment Labeling (TextBlob)**
Sentiment is automatically assigned using **TextBlob** polarity scores:
- Polarity > 0 → **Positive**
- Polarity = 0 → **Neutral**
- Polarity < 0 → **Negative**

**5. Text Cleaning**

**6. Outlier Removal**


**7. Date Filtering**


**8. Translation**


---

## Model Architecture

A separate LSTM model is trained for each user. All three models share the same architecture:

```
Raw Tweet Text
      ↓
Tokenization  (vocabulary size: 5,000 words per user)
      ↓
Sequence Padding  (max length: 56 tokens)
      ↓
Embedding Layer  (5,000 → 128 dimensions)
      ↓
LSTM Layer  (128 units)
      ↓
Dropout Layer  (rate: 0.5)
      ↓
Dense Layer  (128 units, ReLU activation)
      ↓
Output Layer  (3 units, Softmax)
      ↓
Predicted Sentiment: Positive / Neutral / Negative
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | sparse_categorical_crossentropy |
| Epochs | 15 |
| Batch Size | 32 |
| Validation Split | 20% |
| Train/Test Split | 80% / 20% |

---


## How to Use the Trained Models

Open `Model_training_code.ipynb` and run the prediction cells, or use the following code directly in Python:

### Prerequisites

```bash
pip install tensorflow scikit-learn pandas numpy
```

### Load a Model and Predict

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Choose a user
user = "Elon Musk"  # or "ImranKhanPTI" or "cristiano ronaldo"

# Load artifacts for that user
model = load_model(f"trainfiles/{user}_sentiment_model.h5")

with open(f"trainfiles/{user}_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(f"trainfiles/{user}_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Predict sentiment for a new tweet
def predict_sentiment(tweet_text):
    sequence = tokenizer.texts_to_sequences([tweet_text])
    padded = pad_sequences(sequence, maxlen=56, padding="post")
    prediction = model.predict(padded)
    label_index = np.argmax(prediction)
    return label_encoder.inverse_transform([label_index])[0]

# Example
tweet = "This is absolutely incredible news for humanity!"
print(predict_sentiment(tweet))  # Output: Positive
```
---

---



## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | LSTM model training and inference |
| scikit-learn | Label encoding, train/test split |
| TextBlob | Automatic sentiment labeling |
| googletrans | Tweet translation (Portuguese → English) |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Pickle | Saving/loading tokenizers and encoders |
| Jupyter Notebook | Development environment |
