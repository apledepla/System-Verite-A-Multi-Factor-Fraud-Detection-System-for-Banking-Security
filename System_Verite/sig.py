from flask import Flask, render_template, Response, request, jsonify, flash, get_flashed_messages
from flask import Flask, session, redirect, url_for, flash
from ultralytics import YOLO
import cv2
from PIL import Image
import os
from datetime import datetime, timezone
from tensorflow import keras
from keras import layers
import keras
import tensorflow as tf
import numpy as np
import random
import smtplib
from email.message import EmailMessage
import os
import time
import tf_keras
from deepface import DeepFace
from deepface.modules import modeling, verification


# -------------------------------
# EMAIL CONFIG (for OTP)
# -------------------------------
EMAIL_ADDRESS = 'dummybpihacka@gmail.com'
EMAIL_PASSWORD = 'xeui wjmp dttv xodd' 

# -------------------------------
# LOGIN TO HUGGINGFACE & LOAD YOLO SIGNATURE DETECTOR
# -------------------------------

app = Flask(__name__, template_folder='html', static_folder='static')
app.secret_key = os.urandom(24)  # Secret key for session handling

# Download YOLOv8 signature detection model

model = YOLO("yolov8s.pt")


# -------------------------------
# CUSTOM LAYER & LOSS FOR SIGNATURE VERIFICATION (Siamese Network)
# -------------------------------
class EuclideanDist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x1, x2 = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=1, keepdims=True))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

# Contrastive loss for Siamese training
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Load trained Siamese model for signature verification
siamese_model = keras.models.load_model(
    "signet_sia_model.keras",
    custom_objects={
        "EuclideanDist": EuclideanDist,
        "contrastive_loss": contrastive_loss
    }
)

# Load face detector
face_cascade_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam feed
cap = cv2.VideoCapture(0)

# Reference signature path
REFERENCE_PATH = "static/reference.png"

# Globals for storing last detected signature and face
last_cropped_path = None
entire_pic = None


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def load_and_preprocess_image(path):
    """Loads and preprocesses signature image for Siamese model."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, (220, 155))  
    img = img.astype("float32") / 255.0  
    img = np.expand_dims(img, axis=-1)  
    return img

def check_signature(model, path1, path2, threshold=0.5):
    """Compares two signatures using Siamese model."""
    img1 = load_and_preprocess_image(path1)
    img2 = load_and_preprocess_image(path2)
    img1 = np.expand_dims(img1, axis=0)  
    img2 = np.expand_dims(img2, axis=0)
    
    score = model.predict([img1, img2])[0][0]

    result = {
        "similarity_score": float(score),
        "match": "Genuine Match" if score < threshold else "Forged Signature"
    }
    return result


# -------------------------------
# MAIN ROUTES
# -------------------------------

# Default route
@app.route('/')
def index():
    return render_template('general_info.html', image_paths=None)


# -------------------------------
# SIGNATURE CAMERA STREAMING
# -------------------------------
def generate_frames():
    """Streams webcam feed with YOLO signature detection overlay."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
@app.route('/video_feed')
def video_feed():
    """Route for signature detection live stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------
# SIGNATURE & FACE VERIFICATION
# -------------------------------
@app.route('/verify', methods=['POST'])
def verify():
    """Verifies signature (Siamese) and face (DeepFace)."""
    global last_cropped_path

    if not last_cropped_path or not os.path.exists(last_cropped_path):
        return render_template('sign.html', image_paths=None, result="No cropped signature available. Run Detect first.")

    try:
        # Face verification (DeepFace)
        result_face_verification = DeepFace.verify(
            img1_path=f"static/instance_face_photo.png",
            img2_path=f"static/entire_pic.jpg",
        )

        # Signature verification (Siamese model)
        result = check_signature(siamese_model, REFERENCE_PATH, last_cropped_path)
        return render_template(
            'sign.html',
            image_paths=last_cropped_path,
            similarity_score=result["similarity_score"],
            match=result["match"],

            face_verification_match=result_face_verification["verified"],
            face_verification_conf=result_face_verification["confidence"],
            face_verification_distance=result_face_verification["distance"]
        )
    except Exception as e:
        return render_template('sign.html', image_paths=None, result=f"Error: {str(e)}")


# -------------------------------
# FACE CAMERA STREAMING
# -------------------------------
@app.route('/face_cam', methods=['GET', 'POST'])
def face_cam():
    """Loads face camera page."""
    return render_template('face_cam.html')


def adjusted_detect_face(img):
    """Runs Haarcascade face detection."""
    face_img = img.copy()
    face_rect = face_cascade_model.detectMultiScale(
        face_img, scaleFactor=1.2, minNeighbors=5
    )
    return face_rect


def generate_frames_face_detect():
    """Streams webcam with detected faces highlighted."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        face_detect = adjusted_detect_face(frame)

        # Draw rectangles around faces
        for (x, y, w, h) in face_detect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed_face_detect')
def video_feed_face_detect():
    """Route for face detection live stream."""
    return Response(
        generate_frames_face_detect(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/save_face_pic', methods=['POST'])
def save_face_pic():
    """Captures and saves face photo if exactly one face is detected."""
    success, frame = cap.read()
    if not success:
        return jsonify({"status": "error", "message": "Camera not available"}), 200

    face_detect = adjusted_detect_face(frame)

    if len(face_detect) == 1:
        cv2.imwrite("static/instance_face_photo.png", frame)
        return jsonify({
            "status": "success",
            "message": "Photo saved successfully!",
            "image_path": "static/instance_face_photo.png"
        })
    elif len(face_detect) < 1:
        return jsonify({"status": "error", "message": "No face detected"})
    else:
        return jsonify({"status": "error", "message": "More than one face detected"})


# -------------------------------
# SIGNATURE DETECTION
# -------------------------------
@app.route('/detect', methods=['POST'])
def detect():
    """Captures frame, runs YOLO to detect signature, crops it, and validates face presence."""
    global last_cropped_path
    global entire_pic

    success, frame = cap.read()
    if not success:
        return "Camera not available", 500

    # Save entire frame for validation purposes
    entire_pic = "static/entire_pic.jpg"
    cv2.imwrite(entire_pic, frame)

    # Run YOLO detection
    results = model(frame)
    boxes = results[0].boxes
    cropped_path = None

    if boxes is not None:
        for i in range(len(boxes)):
            box = boxes.xyxy[i]
            conf = float(boxes.conf[i])
            if conf >= 0.5:  # threshold
                x1, y1, x2, y2 = map(int, box)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                cropped = pil_image.crop((x1, y1, x2, y2))

                cropped_path = "static/cropped_pic.jpg"
                cropped.save(cropped_path)

                last_cropped_path = cropped_path
                break

    # Run face check on the picture taken
    if entire_pic is not None:
        img = cv2.imread(entire_pic)
        if img is None:
            return render_template('sign.html', result="❌ Error: Could not read image")
        copy_entire = img.copy()
        face_detection = face_cascade_model.detectMultiScale(
            copy_entire, scaleFactor=1.2, minNeighbors=5
        )
        if len(face_detection) != 1:
            return render_template('sign.html', result="❌ No face detected")
        elif len(face_detection) == 1:
            return render_template('sign.html', image_paths=cropped_path, entire_path=entire_pic)


# -------------------------------
# GENERAL INFO & OTP HANDLING
# -------------------------------
@app.route('/general_info', methods=['GET', 'POST'])
def general_info():
    return render_template('general_info.html', image_paths=None)

@app.route('/send_email_otp', methods=['POST'])
def send_email_otp():
    """Sends OTP to given email."""
    email = request.form.get('email')
    if not email:
        flash("Please enter an email address", "error")
        return redirect(url_for('general_info'))

    otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    session['email_otp'] = otp  
    session['email'] = email

    # Gmail SMTP
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

    msg = EmailMessage()
    msg['Subject'] = "OTP Verification"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    msg.set_content(f"Your OTP is: {otp}")

    server.send_message(msg)
    server.quit()

    message = f"OTP sent to Email and Phone Number "
    flash(message, "success")

    return jsonify({"message": message})

@app.route('/verify_email_otp', methods=['POST'])
# Check whether the OTP is correct or not
def verify_email_otp():
    """Verifies entered OTP with session OTP."""
    entered_otp = request.form.get('email_otp')
    if entered_otp == session.get('email_otp'):
        return jsonify({"success": True, "message": "OTP verified successfully!"})
    else:
        return jsonify({"success": False, "message": "Invalid OTP. Please try again."})


# -------------------------------
# SIGNATURE UPLOAD
# -------------------------------
SAVE_PATH = os.path.join("static", "reference.png")

@app.route('/sig_upload', methods=['POST'])
def sig_upload():
    """Loads upload page."""
    return render_template('sig_upload.html')

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles signature file upload and runs YOLO detection."""
    if "image" not in request.files:
        return render_template("sig_upload.html", result="❌ No file part uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("sig_upload.html", result="❌ No file selected")

    # Save uploaded file
    file.save(SAVE_PATH)

    # Run YOLO on uploaded signature
    results = model(SAVE_PATH, conf=0.25)
    detections = results[0].boxes

    # Check whether the uploaded image is a signature or not.
    if len(detections) == 0:
        return render_template("sig_upload.html", result="❌ No signature detected, please upload a valid signature image")

    # Show uploaded image 
    img_url = url_for("static", filename="reference.png") + f"?t={int(time.time())}"
    return render_template("sig_upload.html", image_url=img_url, result="✅ Signature detected successfully!")


# -------------------------------
# RESTART ROUTE
# -------------------------------
@app.route('/restart', methods=['POST'])
def restart():
    """Clears session and restarts app."""
    session.clear()
    return redirect(url_for('general_info'))


if __name__ == '__main__':
    app.run(debug=True)
