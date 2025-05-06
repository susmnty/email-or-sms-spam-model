import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # Tokenize the text into words
    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_model.pkl', 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:  # Check if input is not empty
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict the result
        result = model.predict(vector_input)[0]

        # 4. Display the result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
    else:
        st.warning("Please enter a message to classify.")