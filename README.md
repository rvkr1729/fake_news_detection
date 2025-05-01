                                                     Fake News Detection
This project implements a Fake News Detection system using machine learning techniques. It provides an interactive web application using Streamlit to classify news articles as real or fake.
Project Overview
The Fake News Detection model is based on a Random Forest Classifier and uses various natural language processing (NLP) techniques for feature extraction. It is trained to detect fake news based on the input data and predict whether the news article is real or fake.
Requirements
Before running this project, ensure that you have the following libraries installed:
Python 3.10+
Streamlit
scikit-learn (Version 1.0.2 is recommended for compatibility)
NLTK
pandas
numpy
Install the required dependencies:
pip install -r requirements.txt
 Files 
app.py: Main file that runs the Streamlit app.
fake_news_model_rf.pkl: Pre-trained Random Forest model for fake news detection.
requirements.txt: List of required dependencies
True.csv: Data set with True data points.
Fake.csv:Data set with False data points.
Label_encoder.pkl: Pre trained label encoder.
Tfidf_vectorizer:   The vectorizer initialized and stored.

Requirements.txt:
streamlit==1.15.0
scikit-learn==1.0.2
nltk==3.6.3
pandas==1.3.3
numpy==1.21.6
How to run:
Clone these repository to your local folder.
* git clone https://github.com/yourusername/fake_news_detection.git
* cd fake_news_detection
Install Required Libraries:
pip install -r requirements.txt
Run the  streamlit App:
streamlit run app.py
The app will start, and you can view it in your browser at
·  Local URL: http://localhost:8501
·  Network URL: http://your-ip:8501
How It Works
The app accepts user input in the form of a news article text. The text is preprocessed using NLP techniques (tokenization, stopword removal, etc.), then it is passed through the pre-trained Random Forest model to classify the news article as either real or fake.
Preprocessing Steps:
Tokenization
Removing stopwords
Vectorization using TF-IDF (Term Frequency-Inverse Document Frequency)

Contributing
Feel free to fork this repository, create an issue, or submit a pull request if you'd like to contribute to the project. Make sure to follow these steps to submit a change:
Fork this repository.
Create a new branch (git checkout -b feature-branch).
Make changes and commit them (git commit -m 'Add new feature').
Push to your forked repository (git push origin feature-branch).
Open a pull request.
