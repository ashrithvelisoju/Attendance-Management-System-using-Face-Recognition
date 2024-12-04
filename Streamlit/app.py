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

class AttendanceManagementSystem:
    def __init__(self):
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Ensure necessary directories exist
        os.makedirs('TrainingImage', exist_ok=True)
        os.makedirs('TrainingImageLabel', exist_ok=True)
        os.makedirs('Attendance', exist_ok=True)
        os.makedirs('StudentDetails', exist_ok=True)

    def take_images(self, enrollment, name):
        try:
            cam = cv2.VideoCapture(0)
            sampleNum = 0
            while sampleNum < 70:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    sampleNum += 1
                    cv2.imwrite(f"TrainingImage/{name}.{enrollment}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
                
                st.image(img, channels="BGR", caption=f"Capturing Images: {sampleNum}/70")
                
                if sampleNum >= 70:
                    break
            
            cam.release()
            
            # Save student details to CSV
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            
            with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([enrollment, name, date, time_stamp])
            
            st.success(f"Images saved for Enrollment: {enrollment}, Name: {name}")
        
        except Exception as e:
            st.error(f"Error taking images: {e}")

    def train_model(self):
        try:
            faces, Ids = self.get_images_and_labels("TrainingImage")
            self.recognizer.train(faces, np.array(Ids))
            self.recognizer.save("TrainingImageLabel/Trainner.yml")
            st.success("Model Trained Successfully")
        except Exception as e:
            st.error(f"Error training model: {e}")

    def get_images_and_labels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        Ids = []
        
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(imageNp)
            
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        
        return faceSamples, Ids

    def fill_attendance(self, subject):
        try:
            self.recognizer.read("TrainingImageLabel/Trainner.yml")
            
            cam = cv2.VideoCapture(0)
            df = pd.read_csv("StudentDetails/StudentDetails.csv")
            col_names = ['Enrollment', 'Name', 'Date', 'Time']
            attendance = pd.DataFrame(columns=col_names)
            
            while True:
                ret, im = cam.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    Id, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if conf < 70:
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        aa = df.loc[df['Enrollment'] == Id]['Name'].values
                        tt = f"{Id}-{aa[0]}"
                        
                        attendance.loc[len(attendance)] = [Id, aa[0], date, timeStamp]
                        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 260, 0), 7)
                        cv2.putText(im, tt, (x+h, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
                    
                    else:
                        Id = 'Unknown'
                        tt = Id
                        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 25, 255), 7)
                        cv2.putText(im, tt, (x+h, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 25, 255), 4)
                
                st.image(im, channels="BGR", caption="Attendance Recognition")
                
                if st.button("Stop Recognition"):
                    break
            
            # Save attendance
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour, Minute, Second = timeStamp.split(":")
            
            fileName = f"Attendance/{subject}_{date}_{Hour}-{Minute}-{Second}.csv"
            attendance = attendance.drop_duplicates(['Enrollment'], keep='first')
            attendance.to_csv(fileName, index=False)
            
            st.success("Attendance Filled Successfully")
            st.dataframe(attendance)
        
        except Exception as e:
            st.error(f"Error filling attendance: {e}")

def main():
    st.title("Attendance Management System with Face Recognition")
    ams = AttendanceManagementSystem()
    
    menu = ["Take Images", "Train Model", "Fill Attendance", "View Students"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Take Images":
        st.subheader("Student Image Capture")
        enrollment = st.text_input("Enter Enrollment Number")
        name = st.text_input("Enter Student Name")
        
        if st.button("Capture Images"):
            ams.take_images(enrollment, name)
    
    elif choice == "Train Model":
        st.subheader("Train Face Recognition Model")
        if st.button("Train Model"):
            ams.train_model()
    
    elif choice == "Fill Attendance":
        st.subheader("Fill Attendance")
        subject = st.text_input("Enter Subject Name")
        
        if st.button("Start Face Recognition"):
            ams.fill_attendance(subject)
    
    elif choice == "View Students":
        st.subheader("Registered Students")
        try:
            students_df = pd.read_csv("StudentDetails/StudentDetails.csv")
            st.dataframe(students_df)
        except Exception as e:
            st.error(f"Error reading student details: {e}")

if __name__ == "__main__":
    main()