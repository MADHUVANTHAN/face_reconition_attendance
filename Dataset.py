import cv2
import numpy as np
import os
import pickle

# Open the video stream (camera)
video = cv2.VideoCapture(0)  # Using webcam (0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize face_data list
face_data = []

# Counter for capturing images
i = 0

# Get user name input
name = input("Enter your name: ")

# Check if data directory exists, if not create it
if not os.path.exists('data/'):
    os.makedirs('data/')

# Loop for capturing face images
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize to 50x50 pixels

        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resized_img)  # Append resized face to face_data
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    # Break the loop after capturing 100 faces
    if len(face_data) == 100:
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()

# Save faces in pickle files
face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)  # Flatten the images for saving

# Save names to pickle file
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * 100)  # Append the new names

    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save face data to pickle file
if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)

    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print("Data collection completed successfully!")

