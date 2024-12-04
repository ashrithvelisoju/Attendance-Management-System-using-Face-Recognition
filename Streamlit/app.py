import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import csv
import datetime
import time
from PIL import Image
import pymysql
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class FaceRecognition:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return gray, faces

    def train_model(self, training_path='TrainingImage'):
        def get_images_and_labels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            Ids = []
            
            for imagePath in imagePaths:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = self.face_cascade.detectMultiScale(imageNp)
                
                for (x, y, w, h) in faces:
                    faceSamples.append(imageNp[y:y+h, x:x+w])
                    Ids.append(Id)
            return faceSamples, Ids

        try:
            faces, Ids = get_images_and_labels(training_path)
            self.recognizer.train(faces, np.array(Ids))
            self.recognizer.save("TrainingImageLabel/Trainner.yml")
            return True
        except Exception as e:
            st.error(f"Training failed: {e}")
            return False

    def predict_face(self, gray_frame, face):
        try:
            self.recognizer.read("TrainingImageLabel/Trainner.yml")
            (x, y, w, h) = face
            Id, conf = self.recognizer.predict(gray_frame[y:y+h, x:x+w])
            return Id, conf
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

class DatabaseManager:
    @staticmethod
    def connect_database():
        try:
            connection = pymysql.connect(
                host='localhost', 
                user='root', 
                password='', 
                db='Face_reco_fill'
            )
            return connection
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return None

    @staticmethod
    def create_attendance_table(cursor, table_name):
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ID INT NOT NULL AUTO_INCREMENT,
            ENROLLMENT varchar(100) NOT NULL,
            NAME VARCHAR(50) NOT NULL,
            DATE VARCHAR(20) NOT NULL,
            TIME VARCHAR(20) NOT NULL,
            PRIMARY KEY (ID)
        );
        """
        cursor.execute(sql)

def main():
    st.title("Attendance Management System using Face Recognition")

    menu = ["Home", "Register Student", "Train Model", "Take Attendance", "View Students", "Manually Fill Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)

    face_recognition = FaceRecognition()

    if choice == "Home":
        st.subheader("Welcome to Attendance Management System")
        st.info("Use the sidebar to navigate through different functions")

    elif choice == "Register Student":
        st.subheader("Student Registration")
        enrollment = st.text_input("Enter Enrollment Number")
        name = st.text_input("Enter Name")

        if st.button("Capture Images"):
            if not enrollment or not name:
                st.error("Please enter Enrollment and Name")
            else:
                # Image capture logic here (similar to original take_img function)
                placeholder = st.empty()
                webrtc_ctx = webrtc_streamer(
                    key="face-detection",
                    video_transformer_factory=lambda: VideoTransformer(enrollment, name)
                )

    elif choice == "Train Model":
        st.subheader("Train Face Recognition Model")
        if st.button("Start Training"):
            with st.spinner("Training Model..."):
                success = face_recognition.train_model()
                if success:
                    st.success("Model Trained Successfully!")
                else:
                    st.error("Model Training Failed")

    elif choice == "Take Attendance":
        st.subheader("Automatic Attendance")
        subject = st.text_input("Enter Subject Name")
        
        if st.button("Start Face Recognition"):
            if not subject:
                st.error("Please enter subject name")
            else:
                # Attendance logic here (similar to original Fillattendances function)
                webrtc_ctx = webrtc_streamer(
                    key="attendance",
                    video_transformer_factory=lambda: AttendanceTransformer(subject)
                )

    elif choice == "View Students":
        st.subheader("Registered Students")
        try:
            df = pd.read_csv('StudentDetails/StudentDetails.csv')
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading student details: {e}")

    elif choice == "Manually Fill Attendance":
        st.subheader("Manual Attendance")
        subject = st.text_input("Enter Subject")
        enrollment = st.text_input("Enter Enrollment Number")
        student_name = st.text_input("Enter Student Name")

        if st.button("Add Attendance"):
            if not all([subject, enrollment, student_name]):
                st.error("Please fill all fields")
            else:
                # Manual attendance logic here
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                attendance_data = {
                    'Subject': subject,
                    'Enrollment': enrollment,
                    'Name': student_name,
                    'Date': date,
                    'Time': timestamp
                }
                
                # Save to CSV
                try:
                    with open(f'Attendance/{subject}_manual_attendance.csv', 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=attendance_data.keys())
                        if f.tell() == 0:  # file is empty
                            writer.writeheader()
                        writer.writerow(attendance_data)
                    st.success("Attendance Added Successfully!")
                except Exception as e:
                    st.error(f"Error adding attendance: {e}")

class VideoTransformer(VideoTransformerBase):
    def __init__(self, enrollment, name):
        self.enrollment = enrollment
        self.name = name
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.sample_num = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.sample_num += 1
            
            # Save face images
            cv2.imwrite(f"TrainingImage/{self.name}.{self.enrollment}.{self.sample_num}.jpg", gray[y:y+h, x:x+w])

            if self.sample_num > 70:
                break

        return img

class AttendanceTransformer(VideoTransformerBase):
    def __init__(self, subject):
        self.subject = subject
        self.face_recognition = FaceRecognition()
        self.attendance = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])
        self.students_df = pd.read_csv('StudentDetails/StudentDetails.csv')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr8")
        gray, faces = self.face_recognition.detect_faces(img)

        for face in faces:
            Id, conf = self.face_recognition.predict_face(gray, face)
            
            if Id is not None and conf < 70:
                matching_student = self.students_df[self.students_df['Enrollment'] == Id]
                
                if not matching_student.empty:
                    name = matching_student['Name'].values[0]
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    
                    # Add to attendance if not already present
                    if not ((self.attendance['Enrollment'] == Id) & (self.attendance['Date'] == date)).any():
                        new_entry = pd.DataFrame({
                            'Enrollment': [Id],
                            'Name': [name],
                            'Date': [date],
                            'Time': [timestamp]
                        })
                        self.attendance = pd.concat([self.attendance, new_entry], ignore_index=True)

                    cv2.rectangle(img, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (0, 260, 0), 7)
                    cv2.putText(img, f"{Id}-{name}", (face[0]+face[2], face[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)

        return img

if __name__ == "__main__":
    main()