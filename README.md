# 🛡️ System Verite: A Multi-Factor Fraud Detection System for Banking Security  

**Authors:**  
- Afundar, Audrie Lex L.  
- Khafaji, Mostafa A.  
- Percia, Kyte Daiter M.  
- Rodillas, Christian Miguel T.  

---

## 📖 Introduction  
Intro…  

---

## ✨ Features  

### 📧 E-mail and Phone Number Verification  
- Clients verify their data (E-mail & Phone Number) through One-time passwords (OTPs).  
- Ensures that the stored data are real and valid, reducing fraud risk.  

### 👤 Face Recognition  
- Client’s faces are stored in the database for security and verification purposes.  
- Helps detect fraudulent and scam activities for future transactions on or off site.  

### ✍️ Signature Fraud Detection  
- AI compares two instances of the client’s signature (one given from a previous trusted transaction vs. current transaction).  
- Helps detect forgery and provides another layer for security.  

---

## ⚙️ Installation  
Through GitHub clone the repository or download the folder while retaining the current folder structure.  

Needed dependencies are Python with the following libraries and respective versions:  

- **Flask**: web framework (`flask`) - 3.1.1  
- **Ultralytics YOLO**: object detection (`ultralytics`) - 8.3.170  
- **OpenCV**: computer vision (`cv2`) - 4.12.0  
- **Pillow (PIL)**: image processing (`Pillow`) - 10.4.0  
- **TensorFlow**: deep learning (`tensorflow`) - 2.19.0  
- **Keras**: neural networks (`keras` and `tf-keras`) - 3.10.0  
- **NumPy**: numerical computing (`numpy`) - 2.0.0  
- **DeepFace**: face recognition & analysis (`deepface`) - 0.0.95  

> **Note:** Download through manual install of the dependencies or utilize the `requirements.txt`.  

---

## 🚀 Usage  
To run the app, there are two ways (`app.py`, `sig.py`):  

- **sig.py** – run the Python code through IDE or terminal  
  ```bash
  python sig.py
  python app.py
    
  
System_Verite/
├── .deepface/
├── Html/
│   ├── Face_cam.html
│   ├── General_info.html
│   ├── Sig_upload.html
│   └── Sign.html
├── Static/
│   ├── Css/
│   │   └── style-wr.css
│   ├── Picture/
│   │   ├── Bpi_logo.png
│   │   └── bpi_logo_2.png
│   ├── Cropped_pic.jpg
│   ├── Entire_pic.jpg
│   ├── Instance_face_photo.jpg
│   └── reference.jpg
├── app.py
├── sig.py
├── Haarscascade_frontalface_default.xml
├── Signet_sia_model.keras
└── yolov8s.pt
