import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re

# Streamlit app title
st.title("Email Spam Classifier")

# Sample dataset (Replace with your own data)
data = {
    'email': [
        'Win a $1000 gift card now!',
        'Meeting rescheduled to next Monday',
        'You have won a lottery of $1,000,000',
        'Let us know if you can join the call',
        'Congratulations! You are selected for a free vacation',
        'Can we discuss the project details tomorrow?',
    ],
    'label': ['spam', 'non-spam', 'spam', 'non-spam', 'spam', 'non-spam']
}

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Preprocess the dataset
df = pd.DataFrame(data)
df['email'] = df['email'].apply(preprocess_text)

# Display the dataset in Streamlit
st.write("Dataset:", df)

# Split the dataset into training and testing sets
X = df['email']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Make predictions
y_pred = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Input for new email prediction
st.header("Test Your Own Email")
example_email = st.text_input("Enter an email to classify:")
if example_email:
    preprocessed_email = preprocess_text(example_email)
    example_email_transformed = vectorizer.transform([preprocessed_email])
    prediction = model.predict(example_email_transformed)
    st.write(f"The email is classified as: {prediction[0]}")
