import streamlit as st
import joblib
import numpy as np
from text_loader.loader_new import DataLoader

model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")

loader_new = DataLoader("", False)
loader_new.vectorizer = vectorizer

def get_prediction(input_text):
    try:
        # Step 1: Clean input
        cleaned_text = loader_new.clean_text(input_text)

        # Step 2: Vectorize
        vectorized_text = loader_new.vectorizer.transform([cleaned_text])

        # Step 3: Predict
        pred = model.predict(vectorized_text)[0]
        prob = model.predict_proba(vectorized_text)[0]

        label = "Democrat" if pred == 0 else "Republican"
        confidence = round(float(np.max(prob)), 3)

        return f"{label} (Confidence: {confidence})"

    except Exception as e:
        return {"error": str(e)}       

# Streamlit page configuration
st.set_page_config(page_title="Tweet Classifier", layout="wide")

# Streamlit UI components
st.title("Classify your tweet")

# User inputs the tweet
tweet_input = st.text_input("Enter your tweet", "")

# Button to trigger prediction
if st.button("Classify Tweet"):
    # Get prediction
    prediction = get_prediction(tweet_input)
    
    # Display the prediction
    st.write("Prediction:", prediction)

pass

