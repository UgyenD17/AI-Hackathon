import cv2
import mediapipe as mp
import numpy as np
import os

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Where to save sequences
DATA_DIR = 'skeleton_dataset'  

# Create base folder if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Classes you want to record
actions = ['Correct_Stance', 'Bad_Stance', 'Good_Punch', 'Bad_Punch', 'Good_Kick', 'Bad_Kick']

# Make class folders
for action in actions:
    dir_path = os.path.join(DATA_DIR, action)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Initialize variables
sequence = []  # To accumulate keypoints for the action
images = []  # To accumulate images
label = None
seq_id = 0

def extract_keypoints(results):
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.append(lm.x)
            keypoints.append(lm.y)
        return np.array(keypoints)
    else:
        return np.zeros(66)  # 33 keypoints x (x,y)

# Start webcam
cap = cv2.VideoCapture(0)
print("""
Press keys:
    's' = Correct Stance
    'b' = Bad Stance
    'g' = Good Punch
    'h' = Bad Punch
    'k' = Good Kick
    'l' = Bad Kick
    'q' = Quit
""")

while cap.isOpened():
    ret, frame = cap.read()

    results = pose.process(frame)
    image = frame
    keypoints = extract_keypoints(results)

    # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Skeleton Recorder', image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        label = "Correct_Stance"
        print("Label set: Correct Stance")
    elif key == ord('b'):
        label = "Bad_Stance"
        print("Label set: Bad Stance")
    elif key == ord('g'):
        label = "Good_Punch"
        print("Label set: Good Punch")
    elif key == ord('h'):
        label = "Bad_Punch"
        print("Label set: Bad Punch")
    elif key == ord('k'):
        label = "Good_Kick"
        print("Label set: Good Kick")
    elif key == ord('l'):
        label = "Bad_Kick"
        print("Label set: Bad Kick")

    # If label is set, accumulate keypoints and images
    if label is not None:
        sequence.append(keypoints)
        images.append(image)  # Save the current image

    # Save accumulated keypoints and images when 30 frames have been collected (for example)
    if len(sequence) == 30:  # Adjust the number of frames as needed
        label_path = os.path.join(DATA_DIR, label)
        seq_filename = f"seq_{label}_{seq_id}.npy"
        
        # Save the sequence of keypoints (all 30 frames) into one file
        np.save(os.path.join(label_path, seq_filename), np.array(sequence))
        print(f"Saved {seq_filename} under {label}")

        # Save each image in the sequence as individual files
        for i, img in enumerate(images):
            img_filename = f"seq_{label}_{seq_id}_frame_{i}.jpg"
            cv2.imwrite(os.path.join(label_path, img_filename), img)
            print(f"Saved {img_filename} under {label}")

        # Reset the sequence and images after saving
        sequence = []
        images = []
        seq_id += 1
        label = None  # Reset the label after saving the sequence

cap.release()
cv2.destroyAllWindows()
