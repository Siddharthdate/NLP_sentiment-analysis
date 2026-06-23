import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Sentiment App", layout="centered")

#Custom background (light yellowish white)
st.markdown("""
    <style>
    .stApp {
        background-color: #fff9e6;
    }
    </style>
""", unsafe_allow_html=True)
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

def preprocess_wrapper(texts):
    return [clean_text(t) for t in texts]
# Load pipeline
pipeline = joblib.load("pipeline.pkl")

# Title
st.title(" NLP Sentiment Analysis App")
st.write("Classifies text into Positive, Neutral, or Negative")

# Input
user_input = st.text_area("Enter your text here:")

# Label mapping
label_map = {
    0: "Negative ",
    1: "Neutral ",
    2: "Positive "
}

if st.button(" Predict"):
    if user_input.strip() != "":
        prediction = pipeline.predict([user_input])[0]
        probabilities = pipeline.predict_proba([user_input])[0]

        result = label_map.get(prediction, "Unknown")
        confidence = np.max(probabilities)

        # Result display
        st.markdown("###  Prediction Result")

        if prediction == 2:
            st.success(result)
        elif prediction == 1:
            st.warning(result)
        else:
            st.error(result)

        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        #  Create dataframe for chart
        prob_df = pd.DataFrame({
            "Sentiment": ["Negative ", "Neutral ", "Positive "],
            "Probability": probabilities
        })

        st.markdown("###  Sentiment Distribution")

        #  Bar Chart (clean UI)
        st.bar_chart(prob_df.set_index("Sentiment"))

    else:
        st.warning("Please enter some text")