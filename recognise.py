import cv2
import os
import streamlit as st
import torch
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO

st.title("Face Recognition")


save_folder = "face_database"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cap = cv2.VideoCapture(0)

col1, col2 = st.columns(2)

add_face = col1.checkbox("Add Face")
detect_face = col2.checkbox("Recognise Face")

# Load YOLOv8 model (face detection model)
model = YOLO("yolov8n-face.pt")

db_path = "face_database/"

name = ''
i=0
confid = 0

if(add_face):
    name = st.text_input("Enter your name without spaces:")
    entered = st.checkbox('Name Entered')

FRAME_WINDOW = st.image([])

if add_face and entered:
    i = 0
    if st.button("Capture"):
        i = i + 1

while add_face and entered:

    ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces using YOLOv8
    results = model(frame)

    for result in results:
        for box in result.boxes.xyxy:
            x, y, x1, y1 = map(int, box)
            face = frame[y:y1, x:x1]  # Crop the detected face

            if face.size == 0:
                continue


    FRAME_WINDOW.image(face)

    if i==1:  # Spacebar key to capture image
        img_name = name + '.jpg'
        image_path = os.path.join(save_folder, img_name)
        cv2.imwrite(image_path, face)
        st.write(f"Image saved at: {image_path}")
        i=0

def recognise_frames(frame):
    # Detect faces using YOLOv8
    results = model(frame)

    identities = []

    for result in results:
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            confid = round(float(conf), 2)
            x, y, x1, y1 = map(int, box)
            face = frame[y:y1, x:x1]  # Crop the detected face

            if face.size == 0:
                continue

            try:
                recognition_results = DeepFace.find(img_path=face, db_path=db_path, model_name="ArcFace",
                                                    enforce_detection=False)

                if len(recognition_results[0]) > 0:
                    identity = recognition_results[0]["identity"].values[0].split("/")[-1].split(".")[0]  # Extract name
                else:
                    identity = "Unknown"

            except:
                identity = "Unknown"

            identities.append((x, y, x1, y1, identity, confid))

    return identities

while detect_face:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not ret:
        print("Failed to grab frame")
        break

    recognise_faces = recognise_frames(frame)
    # Recognize the face using ArcFace (DeepFace)


    for (x, y, x1, y1, identity, confidence) in recognise_faces:
        # Draw bounding box and label
        identity_confidence = identity + '  ' + str(confidence)

        if identity == 'Unknown':
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
            cv2.putText(frame, identity_confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            #cv2.putText(frame, str(confidence), (x1, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, identity_confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #cv2.putText(frame, str(confidence), (x1, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    FRAME_WINDOW.image(frame)

