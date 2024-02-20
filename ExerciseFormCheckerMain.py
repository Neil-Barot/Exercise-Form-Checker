import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Confidence threshold: Prevents incorrect calculations from errors in detection, add feature that prompts user to get in frame
confidence_threshold = 0.90

# Form threshold: allowable deviation from proper form
knee_cave_threshold = 0.25 # In degrees
neck_uneven_threshold = 0.35
hip_shift_threshold = 0.1
wrist_uneven_threshold = 0.1
test = 0


# Default Message
message = "Hello World"
issues_found = False
RGB_val = [255, 255, 255]

# Webcam Connection: Use 0 in order to use laptop camera
connection_addr = 0 # "http://(phoneIP):(port)/video"

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
                      mp_pose.PoseLandmark.RIGHT_ANKLE,
                      mp_pose.PoseLandmark.LEFT_HEEL,
                      mp_pose.PoseLandmark.RIGHT_HEEL,
                      mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                      mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

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

def calculate_midpoint(point1, point2):
    # Ensure point1 and point2 are NumPy arrays or convert them if needed
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate midpoint
    midpoint = (point1 + point2) / 2
    return midpoint

def calculate_slope(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    # Ensures that there will never be division by 0. Sacrifices some accuracy
    if x1 == x2:
        x2 += 1

    slope = (y2 - y1) / (x2 - x1)
    return slope

def angle_slope(slope):
    return np.arctan(slope)

def angle_between_lines(slope1, slope2):
    angle_radians = math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2)))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def in_frame(*args):
    # checks whether all of the listed parts are in frame, add if needed
    for arg in args:
        return
    return

# Functions that checks for form issues
def knee_cave(right_hip, left_hip, right_knee, left_knee, right_ankle, left_ankle):
     # Calculates angle of knee joint
        right_knee_angle = landmark_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = landmark_angle(left_hip, left_knee, left_ankle)

        knee_deviation = deviation(right_knee_angle, left_knee_angle)

        if (knee_deviation >= knee_cave_threshold): 
            return True
        else:
            return False

def neck_uneven(right_shoulder, left_shoulder, right_ear, left_ear):
    # Calculates distance between nose and shoulders
    dist_left = landmark_distance(left_shoulder, left_ear)
    dist_right = landmark_distance(right_shoulder, right_ear)

    neck_deviation = deviation(dist_left, dist_right)
    
    if (neck_deviation >= neck_uneven_threshold): 
        return True
    else:
        return False

def hip_shift(right_shoulder, left_shoulder, right_hip, left_hip, nose):
    shoulder_midpoint = calculate_midpoint(right_shoulder, left_shoulder)
    hip_midpoint = calculate_midpoint(right_hip, left_hip)
    #shift = landmark_angle(nose, shoulder_midpoint, hip_midpoint)
    sl = abs(calculate_slope(shoulder_midpoint, hip_midpoint))
    #print(sl)
    slope = angle_slope(sl)
    #print(slope)
    return False
    if (abs(slope) > 300):
        return True
    return False
    
    #hip_deviation = deviation(180, shift)
    #print(slope)
    return
    if (hip_deviation >= hip_shift_threshold): 
        return True
    else:
        return False

def wrist_uneven(right_wrist, left_wrist, right_ear, left_ear):
    # Calculates distance between nose and shoulders
    dist_left = landmark_distance(left_wrist, left_ear)
    dist_right = landmark_distance(right_wrist, right_ear)

    wrist_deviation = deviation(dist_left, dist_right)
    
    if (wrist_deviation >= wrist_uneven_threshold): 
        return True
    else:
        return False

# Open a connection to the webcam
cap = cv2.VideoCapture(connection_addr)

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
        
        nose = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * w,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * h])
        left_ear = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * w,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * h])
        right_ear = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * w,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * h])
        left_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
        right_wrist = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
        left_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
        right_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])


        
        knee_cave_found = knee_cave(right_hip, left_hip, right_knee, left_knee, right_ankle, left_ankle)
        neck_uneven_found = neck_uneven(right_shoulder, left_shoulder, right_ear, left_ear)
        hip_shift_found = hip_shift(right_shoulder, left_shoulder, right_hip, left_hip, nose)
        wrist_uneven_found = wrist_uneven(right_wrist, left_wrist, right_ear, left_ear)

        message = "Form issues: "
        issues_found = False
        if hip_shift_found:
            issues_found = True
            message += "Hip shift, "
        if knee_cave_found:
            issues_found = True
            message += "Knee cave, "
        if neck_uneven_found:
            issues_found = True
            message += "Neck not even, "
        if wrist_uneven_found:
            issues_found = True
            message += "Uneven wrists, "
        if not issues_found:
            message += "No issues found"
            RGB_val = [255, 255, 255]
        else:
            message = message[:-2]
            RGB_val = [50, 50, 255]
            #print(message)

        cv2.putText(frame, f"{message}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (RGB_val[0], RGB_val[1], RGB_val[2]), 1, cv2.LINE_AA)

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
