import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
#from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing import image
import scipy
import os
import cv2

st.title('Emotion Detection')

model = keras.models.load_model('model/my_model')
model.load_weights('model/best_weights.h5')
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
face_haar_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, cap_image = cap.read()
    cap_img_gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)

    faces = face_haar_cascade.detectMultiScale(cap_img_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(cap_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = cap_img_gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = model.predict(img_pixels)
        emotion_label = np.argmax(predictions)

        emotion_prediction = label_dict[emotion_label]

        cv2.putText(cap_image, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    resize_image = cv2.resize(cap_image, (1000, 700))
    st.image(resize_image, channels='BGR')

    if cv2.waitKey(10) == ord('b'):
        break

cap.release()
cv2.destroyAllWindows()
