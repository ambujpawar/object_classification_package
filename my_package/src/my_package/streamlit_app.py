import json
import requests
import urllib.request
import streamlit as st
from PIL import Image


st.title("Predicting the pet breed from an image")

# Take user inputs
img_url = st.text_input("Image URL", "https://www.akc.org/wp-content/uploads/2017/11/Newfoundland-standing-outdoors.jpg")
print(img_url)

st.write("")

# Make a prediction
if st.button("Predict"):
    # Display the image
    img = Image.open(urllib.request.urlopen(img_url))
    st.image(img, use_column_width=True)
    image_url = {"url": img_url}
    response = requests.post("http://127.0.0.1:8000/predict/", json=json.dumps(image_url))
    print(response.json())
    st.write(f"Predicted class: {response.json()['Predicted']}")
