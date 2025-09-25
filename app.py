from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -----------------------------
# Load pre-trained models
# -----------------------------
# Face detector (OpenCV DNN)
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Mask detector (your trained model)
maskNet = load_model("mask_detector.model")

# -----------------------------
# Helper function: detect and predict
# -----------------------------
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = np.expand_dims(face, axis=0) / 255.0

            preds.append(maskNet.predict(face)[0])
            locs.append((startX, startY, endX, endY))

    return (locs, preds)

# -----------------------------
# Video stream generator
# -----------------------------
def generate_frames():
    camera = cv2.VideoCapture(0)  # 0 = default webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

                cv2.putText(frame, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
