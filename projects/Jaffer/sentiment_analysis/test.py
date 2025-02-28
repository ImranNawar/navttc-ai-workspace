import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup
import nltk

# Load the trained models and vectorizer
nb_classifier = joblib.load('nb_classifier_model.pkl')
svm_classifier = joblib.load('svm_classifier_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Get the list of stop words
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize the lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Function to clean and preprocess the input text
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Streamlit UI
st.title("Sentiment Analysis")
st.write("Enter a sentence and click on 'Analyze' to predict the sentiment.")

# Text input
user_input = st.text_input("Enter a sentence:")

# Analyze button
if st.button("Analyze"):
    # Preprocess the input text
    cleaned_input = preprocess_text(user_input)
    
    # Transform the input text using the TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([cleaned_input])
    
    # Predict the sentiment using the Naive Bayes classifier
    nb_prediction = nb_classifier.predict(input_tfidf)
    # Predict the sentiment using the SVM classifier
    svm_prediction = svm_classifier.predict(input_tfidf)
    
    # Display the result
    st.write(f"Naive Bayes classifier: {nb_prediction[0]}")
    st.write(f"SVM classifier: {svm_prediction[0]}")