import json
import requests
import urllib.request
import streamlit as st
from PIL import Image


st.title("Predict pet breed from image")

# https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Staffie.jpg/640px-Staffie.jpg
# https://www.akc.org/wp-content/uploads/2017/11/Newfoundland-standing-outdoors.jpg

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
    response = requests.post("http://127.0.0.1:8000/predict/", json=image_url)
    print(response.json())
    st.write(f"Predicted class: {response.json()['Predicted']}")


if st.button("Predict top 3"):
    # Display the image
    img = Image.open(urllib.request.urlopen(img_url))
    st.image(img, use_column_width=True)
    image_url = {"url": img_url}
    response = requests.post("http://127.0.0.1:8000/predict_top_3/", json=image_url)
    print(response.json())
    st.write(f"Predicted classes: {response.json()['Predicted']}")

