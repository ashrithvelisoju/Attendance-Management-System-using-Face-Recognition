import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import csv

# Paths and constants
TRAINING_IMAGES_PATH = "TrainingImage"
TRAINER_FILE_PATH = "TrainingImageLabel/Trainner.yml"
STUDENT_DETAILS_FILE = "StudentDetails/StudentDetails.csv"
ATTENDANCE_FOLDER = "Attendance"

# Helper functions
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_directory(TRAINING_IMAGES_PATH)
create_directory(ATTENDANCE_FOLDER)

# Function to capture and save student images
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
            image_path = os.path.join(TRAINING_IMAGES_PATH, f"{name}.{enrollment}.{sample_count}.jpg")
            cv2.imwrite(image_path, gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        st.image(img, channels="BGR")
        if sample_count >= 70 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save details to CSV
    with open(STUDENT_DETAILS_FILE, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([enrollment, name, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S')])

    return f"Images captured successfully for {name} with enrollment {enrollment}."

# Train the model using the captured images
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_np = np.array(img, 'uint8')
            id_ = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_np)
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(id_)
        return face_samples, ids

    faces, ids = get_images_and_labels(TRAINING_IMAGES_PATH)
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE_PATH)

    return "Model trained successfully."

# Automatic attendance using face recognition
def automatic_attendance(subject):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE_PATH)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    attendance = pd.DataFrame(columns=["Enrollment", "Name", "Date", "Time"])
    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 70:
                name = f"Student_{id_}"
                date = datetime.now().strftime('%Y-%m-%d')
                time_ = datetime.now().strftime('%H:%M:%S')
                attendance = pd.concat([attendance, pd.DataFrame([[id_, name, date, time_]], columns=attendance.columns)], ignore_index=True)
            else:
                name = "Unknown"

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        st.image(img, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    file_path = os.path.join(ATTENDANCE_FOLDER, f"{subject}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    attendance.to_csv(file_path, index=False)
    return f"Attendance saved to {file_path}"

# Streamlit UI
st.title("Attendance Management System using Face Recognition")

menu = st.sidebar.selectbox("Menu", ["Home", "Capture Images", "Train Model", "Automatic Attendance"])

if menu == "Home":
    st.write("Welcome to the Attendance Management System!")
elif menu == "Capture Images":
    enrollment = st.text_input("Enrollment")
    name = st.text_input("Name")
    if st.button("Capture Images"):
        message = capture_images(enrollment, name)
        st.success(message)
elif menu == "Train Model":
    if st.button("Train Model"):
        message = train_model()
        st.success(message)
elif menu == "Automatic Attendance":
    subject = st.text_input("Subject")
    if st.button("Start Attendance"):
        message = automatic_attendance(subject)
        st.success(message)
