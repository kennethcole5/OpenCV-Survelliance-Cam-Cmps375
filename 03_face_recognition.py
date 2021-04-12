
#!usr/bin/env python

import cv2
import numpy as np
import os
import time
import smtplib

# Email Variables
SMTP_SERVER = 'smtp.gmail.com' #Email Server
SMTP_PORT = 587 #Server Port
GMAIL_USERNAME = 'RaspberryPiSurveyCam@gmail.com'
GMAIL_PASSWORD = 'CMPS375isthebest'

# Create Emailer class
class Emailer:
    def sendmail(self, recipient, subject, content):
          
        # Create Headers
        headers = ["From: " + GMAIL_USERNAME, "Subject: " + subject, "To: " + recipient,
                   "MIME-Version: 1.0", "Content-Type: text/html"]
        headers = "\r\n".join(headers)
  
        # Connect to Gmail Server
        session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        session.ehlo()
        session.starttls()
        session.ehlo()
  
        # Login to Gmail
        session.login(GMAIL_USERNAME, GMAIL_PASSWORD)
  
        # Send Email & Exit
        session.sendmail(GMAIL_USERNAME, recipient, headers + "\r\n\r\n" + content)
        session.quit

sender = Emailer()

# Create recongizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

# Iniciate id counter
id = 0

# Names related to ids: example ==> Kenneth: id=1,  etc
names = ['None', 'Kenneth', 'Raccoon', 'Prof Pao', 'Professor Pao', 'Y'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Create a rectangle over the face and print the ID of the person detected over the rectangle
while True:
    ret, img =cam.read()
#    img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            #id = names[id]
            if(id==4):
                id = names[4]
                sendTo = 'kennyrcole@gmail.com'
                emailSubject = "Pao has been detected!!!"
                emailContent = "Pao was detected at: " +    time.ctime()
                sender.sendmail(sendTo, emailSubject, emailContent)
                print("Email Sent")
            elif(id==1):
                id = names[1]
                print("Student Detected.")
                
            confidence = "  {0}%".format(round(100 - confidence))
                                    
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        # Draw text onto image
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
        
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
