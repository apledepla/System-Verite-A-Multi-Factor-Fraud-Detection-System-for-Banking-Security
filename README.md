# System Verite: A Multi-Factor Fraud Detection System for Banking Security  

**Authors:**  
- Afundar, Audrie Lex L.  
- Khafaji, Mostafa A.  
- Percia, Kyte Daiter M.  
- Rodillas, Christian Miguel T.  

---

## Introduction  

Identity verification serves as the cornerstone of banking security, ensuring that only legitimate individuals can access financial services and conduct transactions. In the Philippine context, where digital transformation is rapidly reshaping how people manage money, the demand for secure authentication methods has grown exponentially. The widespread use of e-wallets, mobile banking, and digital financial platforms has made transactions faster and more convenient, but it has also exposed financial institutions to heightened risks of fraud and identity theft. Traditional safeguards, such as passwords, PIN codes, and manual signature checks, have proven insufficient against increasingly sophisticated attacks by malicious actors.

Moreover, as society becomes more dependent on digital technology, cybercriminals are exploiting vulnerabilities in both human and system processes. While banks and regulators, such as the Bangko Sentral ng Pilipinas (BSP), have implemented new policies like the Anti-Financial Account Scamming Act (AFASA) and Circular 1140, incidents of fraud continue to rise, showing that existing measures may not be enough to address evolving threats. This situation highlights the urgent need for innovation, particularly the integration of artificial intelligence (AI), machine learning, and biometric verification, to provide banks with proactive and intelligent security tools that go beyond traditional, reactive approaches.


---

## Features  

### E-mail and Phone Number Verification  
- Clients verify their data (E-mail & Phone Number) through One-time passwords (OTPs).  
- Ensures that the stored data are real and valid, reducing fraud risk.  

### Face Recognition  
- Client’s faces are stored in the database for security and verification purposes.  
- Helps detect fraudulent and scam activities for future transactions on or off site.  

### Signature Fraud Detection  
- AI compares two instances of the client’s signature (one given from a previous trusted transaction vs. current transaction).  
- Helps detect forgery and provides another layer for security.  

---

## Installation  
Through GitHub clone the repository or download the folder while retaining the current folder structure.  

Download the necessary models and follow the folder structure through the given google drive link:
	https://drive.google.com/drive/folders/1N-8f23Rk4z3k0jRB36Kc5mhQq3wivM5H?usp=sharing

**Python 3.9-3.11** 

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

## Usage  
To run the app, there are two ways (`app.py`, `sig.py`):  

- **sig.py** – run the Python code through IDE or terminal  
  ```bash
  python sig.py
  python app.py

## Folder Structure
```bash
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
```
## Credits & Acknowledgement

GitHub - serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python https://share.google/GjJ5lG8UYYnHNmY0h

Opencv Python program for Face Detection - GeeksforGeeks https://share.google/gfFOHgwz0uMtULPCu

Research inspiration for the Siamese Model:

	SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification - https://arxiv.org/pdf/1707.02131
	Siamese signature verification with confidence - https://www.kaggle.com/code/medali1992/siamese-signature-verification-with-confidence#Contrastive-loss
