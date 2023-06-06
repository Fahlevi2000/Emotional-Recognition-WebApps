import streamlit as st
import numpy as np
import cv2
import keras
from tensorflow.keras.preprocessing import image

st.title('Emotion Detection')

model = keras.models.load_model('model/my_model')
model.load_weights('model/best_weights.h5')
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
face_haar_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_haar_cascade.detectMultiScale(image_gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = image_gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = model.predict(img_pixels)
        emotion_label = np.argmax(predictions)

        emotion_prediction = label_dict[emotion_label]

        cv2.putText(image, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Display the processed image with bounding boxes and emotion labels
    st.image(image, channels='BGR')
