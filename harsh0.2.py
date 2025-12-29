# Importing necessary libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
from collections import deque

pygame.mixer.init()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variables for tracking states
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Buffer to store EAR values for rolling average
ear_buffer = deque(maxlen=6)  # Store EAR of last 6 frames


def play_sound(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def eye_aspect_ratio(eye):
    # Compute EAR using eye landmarks
    A = compute(eye[1], eye[5])
    B = compute(eye[2], eye[4])
    C = compute(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def check_drowsiness(ear_avg):
    global sleep, drowsy, active, status, color
    if ear_avg < 0.21:
        sleep += 1
        drowsy = 0
        active = 0
        if sleep > 6:
            status = "SLEEPING !!!"
            color = (255, 0, 0)
            play_sound("harsh.wav")
    elif 0.21 <= ear_avg <= 0.25:
        sleep = 0
        active = 0
        drowsy += 1
        if drowsy > 6:
            status = "Drowsy !"
            color = (0, 0, 255)
            play_sound("normal.wav")
    else:
        drowsy = 0
        sleep = 0
        active += 1
        if active > 6:
            status = "Active :)"
            color = (0, 255, 0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create two copies of the frame for the different displays
    state_frame = frame.copy()  # This will show the status (Active, Drowsy, Sleeping)
    landmark_frame = frame.copy()  # This will show the landmarks

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Get the eye coordinates
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Compute EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Store the EAR in the buffer
        ear_buffer.append(ear)

        # Calculate the rolling average EAR
        ear_avg = np.mean(ear_buffer)

        # Check drowsiness using the rolling average EAR
        check_drowsiness(ear_avg)

        # Display the status on the "state" frame
        cv2.putText(
            state_frame,
            f"State: {status}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )

        # Display landmarks for visualization on the "landmark" frame
        for x, y in landmarks:
            cv2.circle(landmark_frame, (x, y), 1, (255, 255, 255), -1)

    # Resize both frames to the same height for side-by-side display
    height = frame.shape[0]
    width = frame.shape[1]

    state_frame_resized = cv2.resize(state_frame, (width, height))
    landmark_frame_resized = cv2.resize(landmark_frame, (width, height))

    # Concatenate the two frames horizontally (side by side)
    combined_frame = np.hstack((state_frame_resized, landmark_frame_resized))

    # Show the combined frame
    cv2.imshow("State and Landmarks", combined_frame)

    key = cv2.waitKey(1)
    if key == 27:  # Exit if ESC is pressed
        break

cap.release()
cv2.destroyAllWindows()
