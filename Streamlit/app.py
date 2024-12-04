import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import csv

# Paths
TRAINING_IMAGES_DIR = "TrainingImage"
STUDENT_DETAILS_FILE = "StudentDetails.csv"
TRAINER_FILE = "TrainingImageLabel/Trainer.yml"
ATTENDANCE_DIR = "Attendance"

# Ensure directories exist
os.makedirs(TRAINING_IMAGES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Function to capture student images
def capture_images(enrollment, name):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    sample_count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_count += 1
            img_path = os.path.join(TRAINING_IMAGES_DIR, f"{name}.{enrollment}.{sample_count}.jpg")
            cv2.imwrite(img_path, gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        st.image(img, channels="BGR")
        if sample_count >= 70 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(STUDENT_DETAILS_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([enrollment, name, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S')])

    return f"Images captured successfully for {name} with enrollment {enrollment}."

# Function to train the model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def get_images_and_labels(directory):
        image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
        face_samples, ids = [], []

        for image_path in image_paths:
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            id_ = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(gray_img)
            for (x, y, w, h) in faces:
                face_samples.append(gray_img[y:y+h, x:x+w])
                ids.append(id_)
        return face_samples, ids

    faces, ids = get_images_and_labels(TRAINING_IMAGES_DIR)
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)

    return "Model trained successfully."

# Function for automatic attendance
def mark_attendance(subject):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    attendance = pd.DataFrame(columns=["Enrollment", "Name", "Date", "Time"])
    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            name = f"Student_{id_}" if confidence < 70 else "Unknown"
            if name != "Unknown":
                date = datetime.now().strftime('%Y-%m-%d')
                time_ = datetime.now().strftime('%H:%M:%S')
                attendance.loc[len(attendance)] = [id_, name, date, time_]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        st.image(img, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    file_path = os.path.join(ATTENDANCE_DIR, f"{subject}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    attendance.to_csv(file_path, index=False)

    return f"Attendance saved at {file_path}."

# Streamlit UI
st.title("Attendance Management System")

menu = st.sidebar.selectbox("Menu", ["Home", "Capture Images", "Train Model", "Mark Attendance"])

if menu == "Home":
    st.write("Welcome to the Attendance Management System!")
elif menu == "Capture Images":
    enrollment = st.text_input("Enter Enrollment:")
    name = st.text_input("Enter Name:")
    if st.button("Capture Images"):
        message = capture_images(enrollment, name)
        st.success(message)
elif menu == "Train Model":
    if st.button("Train Model"):
        message = train_model()
        st.success(message)
elif menu == "Mark Attendance":
    subject = st.text_input("Enter Subject Name:")
    if st.button("Mark Attendance"):
        message = mark_attendance(subject)
        st.success(message)
