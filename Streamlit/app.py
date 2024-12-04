import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
# Import your existing functions from main_Run.py here
# For example:
# from main_Run import face_recognition_model, mark_attendance

# Path for storing attendance records
ATTENDANCE_FILE = "attendance.csv"

def load_model():
    # Load your pre-trained LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")  # Ensure this path matches your file
    return recognizer

def detect_and_recognize_face(recognizer):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    stframe = st.empty()
    detected_name = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(face)
            confidence = round(100 - confidence, 2)

            if confidence > 50:  # Adjust confidence threshold as needed
                detected_name = f"Person {id_}"  # Replace with your ID-to-name mapping logic
                cv2.putText(frame, detected_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mark_attendance(detected_name)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display frame in Streamlit
        stframe.image(frame, channels="BGR")

        if detected_name:
            break

    cap.release()
    return detected_name

def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    new_entry = {"Name": name, "Time": current_time}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)

# Streamlit app layout
st.title("Attendance Management System")
st.sidebar.title("Menu")
menu = st.sidebar.radio("Choose an option", ["Home", "Mark Attendance", "View Attendance"])

if menu == "Home":
    st.write("Welcome to the Attendance Management System using Face Recognition!")
    st.image("welcome_image.jpg", caption="Face Recognition")  # Replace with your image
elif menu == "Mark Attendance":
    st.write("Click below to start face recognition:")
    if st.button("Start Face Recognition"):
        recognizer = load_model()
        name = detect_and_recognize_face(recognizer)
        if name:
            st.success(f"Attendance marked for {name}.")
        else:
            st.error("No face detected or recognition failed.")
elif menu == "View Attendance":
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.write("Attendance Records:")
        st.dataframe(df)
    else:
        st.warning("No attendance records found.")
