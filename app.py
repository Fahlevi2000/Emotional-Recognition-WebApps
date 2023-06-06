import streamlit as st
import keras
from PIL import Image
import numpy as np

st.title('Image Classification')

# Load the trained model
model = keras.models.load_model('model/my_model')
model.load_weights('model/best_weights.h5')

# Define the class labels
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((48, 48))  # Resize the image to the input size of the model
    img = img.convert('L')  # Convert image to grayscale
    img = np.array(img)  # Convert image to numpy array
    img = np.expand_dims(img, axis=0)  # Add an extra dimension for batch size
    img = img / 255.0  # Normalize the image
    return img

# Function to make predictions
def make_prediction(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    class_label = label_dict[predicted_class]
    return class_label

# Streamlit app
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the image file and display it
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Make prediction on the uploaded image
    prediction = make_prediction(img)

    # Display the predicted class label
    st.write('Predicted Emotion:', prediction)
