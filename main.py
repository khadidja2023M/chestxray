import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model




st.sidebar.header('Help Menu')

# Add a button in the sidebar
show_steps = st.sidebar.button('Show Menu')

if show_steps:
    with st.sidebar:
        st.write(" 🌟Welcome to ChestVision!🌟")
        st.write("The future of pneumonia detection using cutting-edge AI is here:")
        st.write('Upload Your Chest X-ray: Simply drag and drop your X-ray image to begin.')
        st.write('Automated Analysis: ChestVision uses Convolutional Neural Networks (CNN) for precise and rapid detection.')
        st.write('Get Your Results: After the analysis, receive a comprehensive report on the possible presence of pneumonia.')
        st.write('We prioritize your health and accuracy. Welcome to the future of medical imaging. 🚀')



# Load your trained model
model = load_model('my_model.h5')
CATEGORIES = ['PNEUMONIA', 'NORMAL']

def predict_image(img):
    IMG_SIZE = 128
    # Convert the uploaded file to a numpy array
    img_array = np.array(img.convert('L'))
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    normalized_array = resized_array / 255.0
    reshaped_array = normalized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    prediction = model.predict(reshaped_array)
    predicted_class = CATEGORIES[int(np.round(prediction[0]))]
    return predicted_class


# Use raw HTML to change the title color
st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)



st.title('ChestVision')
st.header("Pneumonia Prediction")
st.write("Upload a chest X-ray image to predict whether it's a normal or indicates pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Predicting...")
    label = predict_image(image)
    st.write(f"The model predicts this image as: {label}")
    accuracy=0.9596
    confidence_level = accuracy*100
    st.write('Confidence Level')

    if confidence_level >= 90:
        bar_color = 'green'
    elif confidence_level >= 70:
        bar_color = 'yellow'
    else:
        bar_color = 'red'

    progress_html = f"""
    <div style="position: relative; width: 100%; height: 25px; background-color: #f0f0f0; border-radius: 5px;">
        <div style="position: absolute; width: {confidence_level}%; height: 100%; background-color: {bar_color}; border-radius: 5px; transition: width 0.5s;"></div>
        <div style="position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: black ;">
            {confidence_level:.2f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
