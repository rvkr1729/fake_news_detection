import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.sparse import hstack

# Download required NLTK data if not already installed

# Load the datasets
fake_d = pd.read_csv("Fake.csv")
true_d = pd.read_csv("True.csv")

# Add label column
fake_d["label"] = 0
true_d["label"] = 1

# Concatenate datasets
df = pd.concat([fake_d, true_d], ignore_index=True)
print(df.shape)
#
# # Remove duplicates and handle missing data
df.drop_duplicates(inplace=True)
print(df.shape)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
# print(df.shape)
# print(df["date"])
print(df.isnull().sum())
df.dropna(inplace=True)
# Clean and preprocess the 'text' column
def clean_word(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    text = re.sub(r"\<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    tokens = word_tokenize(text)

    # Remove short tokens (less than 2 characters)
    tokens = [word for word in tokens if len(word) > 2]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Use stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)
#
# # Apply cleaning function to 'text' column
df['text'] = df['text'].apply(clean_word)

#
# # Check for missing values

# Get unique subjects
df["subject"].unique()

# Split the dataset into features (X) and target (y)
X = df[['text', 'subject']]  # Independent variables
y = df['label']  # Dependent variable
#
# # Train-test split
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# Vectorization for 'text' column
vectorizer = TfidfVectorizer()

# Fit only on training data
X_train_title = vectorizer.fit_transform(X_train['text'])

# Transform test data
X_test_title = vectorizer.transform(X_test['text'])

# Encode 'subject' column
le = LabelEncoder()

# Fit only on training data
X_train_subject = le.fit_transform(X_train['subject']).reshape(-1, 1)

# Transform test data
X_test_subject = le.transform(X_test['subject']).reshape(-1, 1)

# Stack the features horizontally
X_train_final = hstack([X_train_title, X_train_subject])
X_test_final = hstack([X_test_title, X_test_subject])
#
# Model: RandomForestClassifier for better performance
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_final, y_train)

# Predict on the test set
y_pred = model.predict(X_test_final)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model and vectorizer
with open('fake_news_model_rf.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

