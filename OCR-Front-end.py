import streamlit as st
import requests
from PIL import Image
import pytesseract
import base64

def extract_text_from_image(uploaded_file):
    try:
        # Open the image using PIL (Python Imaging Library)
        image = Image.open(uploaded_file)
        
        # Perform OCR using Tesseract with Arabic language
        text = pytesseract.image_to_string(image, lang='ara')
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return None

def classify_text_flask(text):
    try:
        # Send the text to Flask API for classification using POST
        url = "http://localhost:5000/classify"
        data = {'text': text}
        response = requests.post(url, json=data)

        result = response.json()
        return result
    except requests.exceptions.JSONDecodeError:
        st.error("Invalid JSON response from Flask API.")
        return {"comment": "", "hate": 0, "offense": 0}

def main():
    st.title("Arabic Image Text Classification")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Extract text from image using Tesseract OCR
        extraction_result = extract_text_from_image(uploaded_file)

        st.subheader("Text Extraction Result:")
        st.text(extraction_result)

        # Classify extracted text using Flask API
        prediction_result = classify_text_flask(extraction_result)
        print(prediction_result)
        st.subheader("Classification Result:")
        st.text(f"the content of the image  is predicted !: {prediction_result['result']}")
        #st.text(f"Hate: {prediction_result['hate']}")
        #st.text(f"Offense: {prediction_result['offense']}")

if __name__ == "__main__":
    main() 