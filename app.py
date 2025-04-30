import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import scipy.sparse



# Load the saved model, vectorizer, and subject encoder
model = pickle.load(open('fake_news_model_rf.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
subject_encoder = pickle.load(open('label_encoder.pkl', 'rb'))  # You need to have this if you encoded subject

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Define text cleaning function
def clean_word(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    text = re.sub(r"\<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)
# Set background image
# Set background image and custom styles for the text box
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.5);  /* Transparent background */
    border: 2px solid #FFF;  /* White border */
    color: #000;  /* Black text */
    font-size: 16px;
    padding: 10px;
    border-radius: 8px;
}

.stTextArea textarea:focus {
    border: 2px solid #FF6347;  /* Focus border color */
}
</style>
'''

# Add background and custom CSS to Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)



# Streamlit app
st.title("📰 Fake News Detection App")
st.write("Enter a news headline and find out if it's Real or Fake!")

subject = st.selectbox(
    "Choose a subject:",
    ['News', 'politics', 'Government News', 'left-news', 'US_News', 'Middle-east', 'politicsNews', 'worldnews']
)

# User input
user_input = st.text_area("Enter News Title:")

# Prediction Button
if user_input.strip() == "":
    st.warning("⚠️ Please enter some text to analyze.")
if st.button("Predict"):
    with st.spinner('Analyzing...'):
        clean_input = clean_word(user_input)
        if clean_input:
                # 1. Transform text
            transformed_text = vectorizer.transform([clean_input])

                # 2. Encode subject
            encoded_subject = subject_encoder.transform([subject])  # encoded_subject will be array([3]), etc.

                # 3. Reshape subject to sparse format
            subject_sparse = scipy.sparse.csr_matrix(encoded_subject.reshape(-1, 1))

                # 4. Combine both text and subject horizontally
            final_input = scipy.sparse.hstack([transformed_text, subject_sparse])

                # 5. Predict
            prediction = model.predict(final_input)

                # 6. Show result
            if prediction[0] == 1:
                st.success("✅ The news is Real.")
            else:
                st.error("⚠️ The news is Fake!")
        else:
            st.warning("⚠️ Please enter a news title to predict.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
