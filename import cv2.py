import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Keypoints to track for body pose and face
keypoints_to_track = [mp_pose.PoseLandmark.NOSE,
                      mp_pose.PoseLandmark.LEFT_EAR,
                      mp_pose.PoseLandmark.RIGHT_EAR,
                      mp_pose.PoseLandmark.LEFT_SHOULDER,
                      mp_pose.PoseLandmark.RIGHT_SHOULDER,
                      mp_pose.PoseLandmark.LEFT_ELBOW,
                      mp_pose.PoseLandmark.RIGHT_ELBOW,
                      mp_pose.PoseLandmark.LEFT_WRIST,
                      mp_pose.PoseLandmark.RIGHT_WRIST,
                      mp_pose.PoseLandmark.LEFT_HIP,
                      mp_pose.PoseLandmark.RIGHT_HIP,
                      mp_pose.PoseLandmark.LEFT_KNEE,
                      mp_pose.PoseLandmark.RIGHT_KNEE,
                      mp_pose.PoseLandmark.LEFT_ANKLE,
                      mp_pose.PoseLandmark.RIGHT_ANKLE]

def landmark_distance(point1, point2):
    return np.linalg.norm(point2 - point1)

def landmark_angle(point1, center_point, point2):
    vector1 = point1 - center_point
    vector2 = point2 - center_point
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def deviation(val1, val2):
    return np.abs(val1 - val2) / ((val1 + val2) / 2)

# Confidence threshold: Prevents incorrect calculations from errors in detection, add feature that prompts user to get in frame
confidence_threshold = 0.90

# Form threshold: allowable deviation from proper form
knee_cave_threshold = 0.1

# Default Message
message = "Hello World"

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        for i, landmark in enumerate(keypoints_to_track):
            landmark_point = results.pose_landmarks.landmark[landmark]
            h, w, c = frame.shape
            cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
            
            # Display landmark and confidence score if above threshold
            confidence = landmark_point.visibility
            if confidence > confidence_threshold:
                cv2.putText(frame, f'{i}: {confidence:.2f}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            

        # Positions of landmarks
        left_hip = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                             results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * h])
        right_hip = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * h])
        left_knee = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * h])
        right_knee = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * h])
        left_ankle = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * h])
        right_ankle = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h])
        
        # Calculates angle of knee joint
        right_knee_angle = landmark_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = landmark_angle(left_hip, left_knee, left_ankle)

        knee_deviation = deviation(right_knee_angle, left_knee_angle)
        # Display messages at the bottom of the window

        if (knee_deviation >= knee_cave_threshold): 
            message = "Knee Cave Detected"
        else:
            message = "No Form Errors"

        cv2.putText(frame, f"{message}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display Knee angles
        #cv2.putText(frame, f'Right Knee: {right_knee:.2f} degrees | Left Knee: {left_knee:.2f} degrees', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Mediapipe Body and Face Pose Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()
