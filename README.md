## 📧 SMS/Email Spam Detection using Machine Learning

This project is a simple and effective **Spam Detection System** built using **Machine Learning** techniques. It classifies incoming messages (SMS or email) as either **Spam** or **Ham (Not Spam)** based on their content. The model is trained on a labeled dataset of sample messages using **TF-IDF Vectorization** and a **Multinomial Naive Bayes** classifier.

### 🔍 Problem Statement
Spam messages can clutter inboxes, waste time, and even pose security risks. This project aims to build a lightweight machine learning model that can automatically detect and filter such spam messages in real time.

---

### 🧠 Technologies Used

- **Python** 🐍
- **Scikit-learn** (for ML algorithms)
- **TfidfVectorizer** (for converting text to numerical format)
- **Multinomial Naive Bayes** (a probabilistic classifier ideal for text data)
- **Streamlit** (for building a user-friendly web app interface)
- **Pickle** (to save and load the model/vectorizer)

---

### 🛠️ How it Works

1. **Data Preparation**: A dataset containing labeled messages is used where `1 = Spam` and `0 = Ham`.
2. **Text Preprocessing**: Messages are converted into numerical vectors using **TF-IDF**.
3. **Model Training**: A **Multinomial Naive Bayes** classifier is trained on the vectorized data.
4. **Model Saving**: The trained model and vectorizer are saved using `pickle`.
5. **Prediction Interface**: A simple **Streamlit** UI allows users to input a message and instantly check whether it’s spam or not.

---

### 🚀 How to Run

1. Clone this repository.
2. Run `train.py` to train and save the model.
3. Run `app.py` with Streamlit:
   ```bash
   streamlit run app.py
   ```

---

### 📈 Future Improvements

- Use a larger and real-world dataset (e.g., from UCI repository).
- Include preprocessing like stemming, stopword removal, etc.
- Try deep learning models like LSTM for better accuracy.
- Deploy the app using Streamlit Cloud or Heroku.
