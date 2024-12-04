import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import csv
import datetime
import time
import pymysql


class AttendanceManagementSystem:
    def __init__(self):
        # Initialize session state variables
        if 'page' not in st.session_state:
            st.session_state.page = 'Home'
        
        # Initialize face detector
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def load_student_details(self):
        """Load student details from CSV"""
        try:
            return pd.read_csv('StudentDetails/StudentDetails.csv')
        except FileNotFoundError:
            st.error("Student details file not found!")
            return pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])

    def take_images(self, enrollment, name):
        """Capture face images for a student"""
        try:
            # Ensure TrainingImage directory exists
            os.makedirs('TrainingImage', exist_ok=True)
            
            # Open camera
            cap = cv2.VideoCapture(0)
            sample_num = 0
            
            st.write("Press 'q' to stop capturing")
            
            while sample_num < 70:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Save the captured face
                    sample_num += 1
                    cv2.imwrite(f"TrainingImage/{name}.{enrollment}.{sample_num}.jpg", gray[y:y+h, x:x+w])
                
                # Display the frame
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')
                
                # Break condition
                if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 70:
                    break
            
            cap.release()
            
            # Save student details to CSV
            now = datetime.datetime.now()
            with open('StudentDetails/StudentDetails.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([enrollment, name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])
            
            st.success(f"Images saved for {name} with Enrollment {enrollment}")
        
        except Exception as e:
            st.error(f"Error capturing images: {e}")

    def train_model(self):
        """Train face recognition model"""
        try:
            # Ensure TrainingImageLabel directory exists
            os.makedirs('TrainingImageLabel', exist_ok=True)
            
            # Create face recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Get images and labels
            faces, ids = self._get_images_and_labels('TrainingImage')
            
            # Train the model
            recognizer.train(faces, np.array(ids))
            
            # Save the trained model
            recognizer.save('TrainingImageLabel/Trainner.yml')
            
            st.success("Model trained successfully!")
        
        except Exception as e:
            st.error(f"Error training model: {e}")

    def _get_images_and_labels(self, path):
        """Get images and corresponding labels for training"""
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        Ids = []
        
        for imagePath in imagePaths:
            # Load image and convert to grayscale
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            
            # Extract ID from filename
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            
            # Detect faces
            faces = self.detector.detectMultiScale(imageNp)
            
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        
        return faceSamples, Ids

    def fill_attendance(self, subject):
        """Fill attendance using face recognition"""
        try:
            # Load trained model
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("TrainingImageLabel/Trainner.yml")
            
            # Load student details
            df = pd.read_csv("StudentDetails/StudentDetails.csv")
            
            # Open camera
            cap = cv2.VideoCapture(0)
            
            # Prepare attendance DataFrame
            col_names = ['Enrollment', 'Name', 'Date', 'Time']
            attendance = pd.DataFrame(columns=col_names)
            
            start_time = time.time()
            detection_duration = 20  # 20 seconds of detection
            
            st.write("Attendance recognition in progress...")
            
            while time.time() - start_time < detection_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detector.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    # Recognize face
                    Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if conf < 70:
                        # Find name from student details
                        name = df.loc[df['Enrollment'] == Id, 'Name'].values
                        
                        # Add to attendance
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        new_entry = pd.DataFrame({
                            'Enrollment': [Id],
                            'Name': name,
                            'Date': [date],
                            'Time': [timestamp]
                        })
                        
                        attendance = pd.concat([attendance, new_entry], ignore_index=True)
                
                # Display frame (optional)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')
            
            cap.release()
            
            # Remove duplicates
            attendance = attendance.drop_duplicates(subset=['Enrollment'])
            
            # Save attendance to CSV
            filename = f"Attendance/{subject}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            os.makedirs('Attendance', exist_ok=True)
            attendance.to_csv(filename, index=False)
            
            st.success("Attendance filled successfully!")
            st.dataframe(attendance)
            
            return attendance
        
        except Exception as e:
            st.error(f"Error filling attendance: {e}")

    def manually_fill_attendance(self, subject):
        """Manually fill attendance"""
        st.header("Manually Fill Attendance")
        
        # Create input fields
        enrollment = st.text_input("Enter Enrollment Number")
        student_name = st.text_input("Enter Student Name")
        
        if st.button("Add Student to Attendance"):
            try:
                # Prepare attendance entry
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                # Create attendance entry
                attendance_entry = pd.DataFrame({
                    'Enrollment': [enrollment],
                    'Name': [student_name],
                    'Date': [date],
                    'Time': [timestamp],
                    'Subject': [subject]
                })
                
                # Save to CSV
                filename = f"Attendance/{subject}_{date}.csv"
                os.makedirs('Attendance', exist_ok=True)
                
                # Append to existing file or create new
                try:
                    existing = pd.read_csv(filename)
                    updated = pd.concat([existing, attendance_entry], ignore_index=True)
                except FileNotFoundError:
                    updated = attendance_entry
                
                updated.to_csv(filename, index=False)
                
                st.success(f"Added {student_name} to attendance")
                st.dataframe(updated)
            
            except Exception as e:
                st.error(f"Error adding student: {e}")

    def admin_login(self):
        """Admin login and student details view"""
        st.header("Admin Login")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == 'abc' and password == 'abc':
                # Load and display student details
                try:
                    students = pd.read_csv('StudentDetails/StudentDetails.csv')
                    st.subheader("Registered Students")
                    st.dataframe(students)
                except FileNotFoundError:
                    st.error("No student details found!")
            else:
                st.error("Incorrect Username or Password")

    def main(self):
        """Main application interface"""
        st.title("Attendance Management System")
        
        # Sidebar navigation
        menu = ["Home", "Take Images", "Train Model", "Fill Attendance", 
                "Manually Fill Attendance", "Admin Login"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Home":
            st.header("Welcome to Attendance Management System")
            st.write("Use the sidebar to navigate through different functions.")
        
        elif choice == "Take Images":
            st.header("Take Student Images")
            enrollment = st.text_input("Enter Enrollment Number")
            name = st.text_input("Enter Student Name")
            
            if st.button("Capture Images"):
                if enrollment and name:
                    self.take_images(enrollment, name)
                else:
                    st.warning("Please enter Enrollment and Name")
        
        elif choice == "Train Model":
            st.header("Train Face Recognition Model")
            if st.button("Start Training"):
                self.train_model()
        
        elif choice == "Fill Attendance":
            st.header("Fill Attendance")
            subject = st.text_input("Enter Subject Name")
            
            if st.button("Start Face Recognition"):
                if subject:
                    self.fill_attendance(subject)
                else:
                    st.warning("Please enter subject name")
        
        elif choice == "Manually Fill Attendance":
            st.header("Manually Fill Attendance")
            subject = st.text_input("Enter Subject Name")
            
            if subject:
                self.manually_fill_attendance(subject)
        
        elif choice == "Admin Login":
            self.admin_login()

def main():
    ams = AttendanceManagementSystem()
    ams.main()

if __name__ == "__main__":
    main()