import cv2
import mediapipe as mp
import numpy as np
 
# Initialize mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
 
# Custom drawing settings for body landmarks
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5)
connection_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2)
 
# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
 
video_path = "/Users/ut/downloads/IMG_9905.MOV"
  # Update path after converting video
cap = cv2.VideoCapture(video_path)
 
if not cap.isOpened():
    print(f"Error: Couldn't open the video file at {video_path}")
else:
    print(f"Successfully opened the video file: {video_path}")
 
# Set up Pose model for analysis
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
 
    frame_count = 0
    punch_counter = 0
    kick_counter = 0
 
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty video frame.")
            break
 
        # Make image read-only for performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        # Process the image and find pose
        results = pose.process(image)
 
        # Make image writable to draw landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )
 
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
 
            # Get joints for calculations
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
 
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
 
            # Calculate angles for detecting punches/kicks
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
 
            # Example rules to detect punches/kicks:
            if elbow_angle < 45:  # Detect punch if elbow angle is sharp
                punch_counter += 1
            if knee_angle < 60:   # Detect kick if knee angle is sharp
                kick_counter += 1
 
        # Draw text before flipping the image to prevent inverted text
        cv2.putText(image, f'Punches: {punch_counter}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text for Punches
        cv2.putText(image, f'Kicks: {kick_counter}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text for Kicks
 
        # Remove or comment this line to avoid flipping the image
        # flipped_image = cv2.flip(image, 1)  # This line is removed
 
        # Show the non-flipped image
        cv2.imshow('UFC Fighter Body Tracking', image)  # Display the original image
 
        # Exit the loop when ESC key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break
 
# Release resources
cap.release()
cv2.destroyAllWindows()