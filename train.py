import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# Sample training data
data = {
    'text': [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now.",
        "Hey, are we still meeting for lunch today?",
        "Win cash now! Click here to claim your prize.",
        "Hi Mom, I'll call you later tonight.",
        "URGENT! You have won a 1 week FREE membership!",
        "Don't forget about the meeting tomorrow morning.",
        "Congratulations! You have been selected for a prize. Call now!",
        "Are you coming to the party tonight?",
        "You won a lottery! Send your bank details now.",
        "Let's catch up this weekend."
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train-test split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer (Refitting here)
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)  # Fit the vectorizer on the training data

# Naive Bayes Model (Same as before)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the fitted vectorizer and model

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save the trained model
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Training complete! Files 'vectorizer.pkl' and 'spam_model.pkl' saved.")
