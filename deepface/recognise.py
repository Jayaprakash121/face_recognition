from deepface import DeepFace
import cv2
import os

cap = cv2.VideoCapture('video3_hd.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Face recognition
        recognition_results = DeepFace.find(
            img_path=frame,
            db_path="database/",
            model_name="ArcFace",
            detector_backend='yolov8',
            enforce_detection=False
        )

        # Get face locations separately (bounding boxes)
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend='yolov8',
            enforce_detection=False
        )

        if isinstance(recognition_results, list):
            for i, face_info in enumerate(faces):
                if i >= len(recognition_results):
                    break
                df = recognition_results[i]
                if df.empty:
                    name = "Unknown"
                else:
                    identity_path = df.iloc[0]["identity"]
                    distance = round(df.iloc[0]["distance"], 3)
                    name = os.path.basename(identity_path).split('.')[0]
                    name = f"{name} ({distance})"

                region = face_info["facial_area"]
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except Exception as e:
        print(f"[!] Frame error: {e}")
        continue

    cv2.imshow("DeepFace Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
