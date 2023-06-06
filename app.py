import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np
import cv2
import keras
from tensorflow.keras.preprocessing import image

st.title('Emotion Detection')

model = keras.models.load_model('model/my_model')
model.load_weights('model/best_weights.h5')
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
face_haar_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        super().__init__()

    def recv(self, frame):
        cap_image = frame.to_ndarray(format="bgr24")
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

        return cap_image

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetector,
    async_processing=True,
    desired_playing_width=640,
    # Add other parameters as needed
)

if webrtc_ctx.video_processor:
    st.video(webrtc_ctx.video_processor.frame_out)

