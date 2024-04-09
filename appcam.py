import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import cv2
from PIL import Image

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    text = text.lower()  
    text = word_tokenize(text)  
    text = [word for word in text if word.isalnum()]  
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]  
    text = [ps.stem(word) for word in text]  
    return " ".join(text)

# Define function to check human presence
# Define function to check human presence using Haar Cascade Classifier
# Define function to check human presence using Haar Cascade Classifier
def check_human_presence():
    # Load the Haar Cascade Classifier for human detection
    human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # OpenCV code to activate camera
    cap = cv2.VideoCapture(0)
    count_no_human = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect human faces in the frame
        faces = human_cascade.detectMultiScale(gray, 1.1, 4)

        # If human faces are detected, reset the count
        if len(faces) > 0:
            count_no_human = 0

        # Check if no human faces are detected
        if len(faces) == 0:
            count_no_human += 1
            if count_no_human == 5:
                st.error("Alert: No human detected. Exam needs to be stopped.")
                break

    cap.release()



# Define the Streamlit app
def main():
    st.title("Exam")

    for i in range(3):
        # Text input
        text_input = st.text_area(f"Enter your text {i+1} here:", 'what is your name?')

        if st.button(f"Classify {i+1}"):
            if text_input:
                # Preprocess the input text
                preprocessed_text = preprocess_text(text_input)

                # Vectorize the preprocessed text
                text_vectorized = tfidf_vectorizer.transform([preprocessed_text])

                # Make predictions
                prediction = model.predict(text_vectorized)[0]

                # Display the prediction
                if prediction == 1:
                    st.success(f"Prediction {i+1}: Correct")
                else:
                    st.error(f"Prediction {i+1}: Wrong")
            else:
                st.warning("Please enter some text.")

    # Check for human presence
    check_human_presence()

if __name__ == "__main__":
    main()
