import cv2
import os
from ultralytics import YOLO
from deepface import DeepFace
import time

db_path = "new_database/"

cap = cv2.VideoCapture('video3_hd.mp4')
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
model = YOLO("yolov11m-face.pt").to('cuda')

ct=0
pt=0
while True:
    #cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    ct = time.time()
    fps = 1 / (ct - pt)
    pt = ct

    for result in results:
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            confid = round(float(conf), 2)
            x, y, x1, y1 = map(int, box)

            padding = 30
            x = x - padding
            y = y - padding
            x1 = x1 + padding
            y1 = y1 + padding

            face = frame[y:y1, x:x1]  # Crop the detected face

            if face.size == 0:
                continue

            identity = ' '

            recognition_results = DeepFace.find(
                img_path=face,
                db_path=db_path,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend='yolov8',
                align=True,
                threshold=0.65,
                normalization='ArcFace'
            )

            face_rec_conf = 0
            if len(recognition_results[0]) > 0:
                face_rec_conf = recognition_results[0]["distance"].values[0]
                if face_rec_conf < 0.65:
                    identity = recognition_results[0]["identity"].values[0].split("/")[-1].split(".")[0]
                else:
                    identity = "Unknown"
            #        break

            for folder in os.listdir(db_path):
                index = identity.find(folder)
                if index != -1:
                    identity = folder
                    break

            if identity == ' ':
                identity = 'Unknown'
            
            identity_confidence = identity + '  ' + str(round(float(face_rec_conf), 3))
            if identity == 'Unknown':
                cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
                cv2.putText(frame, identity_confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, identity_confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imshow("Live Face Recognition - YOLOv11m_ArcFace", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()