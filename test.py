import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe and OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Path to the video file
video_path = "/Users/ut/Desktop/IMG_3267.MOV"

# Constants for punch and kick detection
GOOD_PUNCH_ELBOW = 180      
GOOD_SHOULDER_ANGLE = 70    
GOOD_KICK_KNEE_ANGLE_MIN = 50  
GOOD_KICK_KNEE_ANGLE_MAX = 150  
ANGLE_TOLERANCE = 30        
KICK_DEVIATION = 30         

# Idle time threshold (in seconds)
IDLE_THRESHOLD = 2.0  

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Setup Mediapipe Pose instance
cap = cv2.VideoCapture(0)  # Open the video file

# Time tracking for idle state
last_move_time = cv2.getTickCount()  
in_fighting_stance = True  
prev_left_elbow_angle = None
prev_right_elbow_angle = None
prev_left_knee_angle = None
prev_right_knee_angle = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Rotate frame if width > height (i.e., video is horizontal)
        frame_height, frame_width = frame.shape[:2]
        # if frame_width > frame_height:
        #     frame = cv2.rotate(frame, cv2.ROTATE__COUNTERCLOCKWISE)

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Arm coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Leg coordinates
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_shoulder_hip_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            right_shoulder_hip_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)  # New for kicks
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)  # New for kicks

            # Movement detection
            movement_detected = False
            if prev_left_elbow_angle and abs(left_elbow_angle - prev_left_elbow_angle) > 5:
                movement_detected = True
            if prev_right_elbow_angle and abs(right_elbow_angle - prev_right_elbow_angle) > 5:
                movement_detected = True
            if prev_left_knee_angle and abs(left_knee_angle - prev_left_knee_angle) > 5:
                movement_detected = True
            if prev_right_knee_angle and abs(right_knee_angle - prev_right_knee_angle) > 5:
                movement_detected = True
            
            if movement_detected:
                last_move_time = cv2.getTickCount()

            # Idle vs Moving
            if (cv2.getTickCount() - last_move_time) / cv2.getTickFrequency() > IDLE_THRESHOLD:
                in_fighting_stance = True
            else:
                in_fighting_stance = False

            # Display stance
            if in_fighting_stance:
                if (abs(left_elbow_angle - GOOD_PUNCH_ELBOW) <= ANGLE_TOLERANCE and 
                    abs(left_shoulder_hip_angle - GOOD_SHOULDER_ANGLE) <= ANGLE_TOLERANCE):
                    stance = "Good Fighting Stance"
                else:
                    stance = "Bad Fighting Stance"
                
                cv2.putText(image, stance, (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Punch detection
            if (abs(left_elbow_angle - GOOD_PUNCH_ELBOW) <= ANGLE_TOLERANCE and 
                abs(left_shoulder_hip_angle - GOOD_SHOULDER_ANGLE) <= ANGLE_TOLERANCE):
                punch_type = "Good Left Punch"
            elif (abs(right_elbow_angle - GOOD_PUNCH_ELBOW) <= ANGLE_TOLERANCE and 
                  abs(right_shoulder_hip_angle - GOOD_SHOULDER_ANGLE) <= ANGLE_TOLERANCE):
                punch_type = "Good Right Punch"
            else:
                punch_type = "Bad Punch"

            # Kick detection
            if (GOOD_KICK_KNEE_ANGLE_MIN <= left_knee_angle <= GOOD_KICK_KNEE_ANGLE_MAX and 
                GOOD_KICK_KNEE_ANGLE_MIN <= left_hip_angle <= GOOD_KICK_KNEE_ANGLE_MAX):
                kick_type = "Good Left Kick"
            elif (GOOD_KICK_KNEE_ANGLE_MIN <= right_knee_angle <= GOOD_KICK_KNEE_ANGLE_MAX and 
                  GOOD_KICK_KNEE_ANGLE_MIN <= right_hip_angle <= GOOD_KICK_KNEE_ANGLE_MAX):
                kick_type = "Good Right Kick"
            else:
                kick_type = "Bad Kick"

            # Display punch and kick types
            cv2.putText(image, f"{punch_type}", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Good" in punch_type else (0, 0, 255), 
                        2, cv2.LINE_AA)
            cv2.putText(image, f"{kick_type}", (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Good" in kick_type else (0, 0, 255), 
                        2, cv2.LINE_AA)

            # Update previous angles
            prev_left_elbow_angle = left_elbow_angle
            prev_right_elbow_angle = right_elbow_angle
            prev_left_knee_angle = left_knee_angle
            prev_right_knee_angle = right_knee_angle

        except Exception as e:
            print(f"Error processing frame: {e}")
        
        # Render landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=8), 
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=8))
        
        # Show image
        cv2.imshow('Pose Detection', image)

        # SLOW MOTION EFFECT: wait 50 ms per frame (instead of 1 ms)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
