import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ======================
# CONFIG
# ======================
MODEL_PATH = "emotion_cnn_small_dataset.h5"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMG_SIZE = 48

EMOTIONS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprised",
    "Neutral"
]

# ======================
# LOAD MODEL & FACE DETECTOR
# ======================
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# ======================
# SMOOTHING BUFFER
# ======================
prediction_buffer = deque(maxlen=10)
last_emotion = "Detecting..."
last_confidence = 0.0

# ======================
# START WEBCAM
# ======================
cap = cv2.VideoCapture(0)

print("🎥 Webcam started — press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        # Show last emotion even if face is lost
        cv2.putText(
            frame,
            f"{last_emotion} ({last_confidence*100:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        preds = model.predict(face, verbose=0)[0]
        prediction_buffer.append(preds)

        avg_preds = np.mean(prediction_buffer, axis=0)
        emotion_idx = np.argmax(avg_preds)

        last_emotion = EMOTIONS[emotion_idx]
        last_confidence = avg_preds[emotion_idx]

        label = f"{last_emotion} ({last_confidence*100:.1f}%)"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stopped")