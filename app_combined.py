import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter
import math


model = tf.keras.models.load_model('final_baro_model.h5')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_vertical_distance_cm(landmark1, landmark2, frame_height, distance_to_camera_cm=60, camera_fov_degrees=25):
    if landmark1 is None or landmark2 is None:
        return None

    landmark1_pixel = landmark1[1] * frame_height
    landmark2_pixel = landmark2[1] * frame_height

    pixel_distance = np.abs(landmark1_pixel - landmark2_pixel)

    real_height_cm = 2 * distance_to_camera_cm * np.tan(np.radians(camera_fov_degrees / 2))
    cm_per_pixel = real_height_cm / frame_height
    vertical_distance_cm = pixel_distance * cm_per_pixel

    return vertical_distance_cm

def calculate_angle(p1, p2):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle
    return angle

def adjust_angle(angle):
    if angle > 180:
        angle = 360 - angle
    return angle

def evaluate_angle_condition(angle):
    adjusted_angle = adjust_angle(angle)

    if 165 <= adjusted_angle <= 180:
        return 'Fine'
    elif 150 <= adjusted_angle < 165:
        return 'Danger'
    elif 135 <= adjusted_angle < 150:
        return 'Serious'
    elif adjusted_angle < 135:
        return 'Very Serious'

def extract_frames(video_file, interval=5):
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)
    images = []
    landmarks_info = []
    angle_conditions = []

    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % (frameRate * interval) == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_frame, (28, 28))
            normalized_img = resized_img / 255.0
            normalized_img = np.stack((normalized_img,)*3, axis=-1)
            images.append(normalized_img)

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y]
                
                vertical_distance_cm = calculate_vertical_distance_cm(left_shoulder, left_ear, frame.shape[0])
                angle = calculate_angle(left_ear, left_shoulder)
                adjusted_angle = adjust_angle(angle)
                angle_status = evaluate_angle_condition(adjusted_angle)

                landmarks_info.append((left_shoulder, left_ear, vertical_distance_cm, adjusted_angle))
                angle_conditions.append(angle_status)

    status_frequencies = Counter(angle_conditions)
    cap.release()
    return np.array(images), landmarks_info, dict(status_frequencies)

# def calculate_posture_ratios(predictions):
#     hunched_posture_label = 0
#     normal_posture_label = 1

#     total_predictions = len(predictions)
#     hunched_count = np.sum(predictions == hunched_posture_label)
#     normal_count = np.sum(predictions == normal_posture_label)

#     hunched_ratio = (hunched_count / total_predictions) * 100
#     normal_ratio = (normal_count / total_predictions) * 100

#     return hunched_ratio, normal_ratio


def calculate_posture_ratios_based_on_angle(conditions):
    fine_or_normal = ['Fine', 'Danger']
    hunched_conditions = ['Serious', 'Very Serious']

    total_conditions = len(conditions)
    normal_count = sum(1 for cond in conditions if cond in fine_or_normal)
    hunched_count = sum(1 for cond in conditions if cond in hunched_conditions)

    normal_ratio = (normal_count / total_conditions) * 100
    hunched_ratio = (hunched_count / total_conditions) * 100

    return hunched_ratio, normal_ratio


def calculate_scores(predictions_proba):
    scores = np.max(predictions_proba, axis=1) * 100  
    return scores.tolist()

def analyze_video(video_path):
    images, landmarks_info, status_frequencies = extract_frames(video_path)

    angle_conditions = [evaluate_angle_condition(info[3]) for info in landmarks_info]
    hunched_ratio, normal_ratio = calculate_posture_ratios_based_on_angle(angle_conditions)

    return {
        'hunched_ratio': hunched_ratio,
        'normal_ratio': normal_ratio,
        'landmarks_info': [
            {
                'left_shoulder': {'x': info[0][0], 'y': info[0][1]},
                'left_ear': {'x': info[1][0], 'y': info[1][1]},
                'vertical_distance_cm': info[2],
                'angle': info[3]
            } for info in landmarks_info
        ],
        'status_frequencies': status_frequencies
    }

st.title("Posture Video Analysis")

video_file_path = st.text_input("Enter the path to the video file for analysis:")

view_option = st.radio("Choose result view:", ('Normal/Hunched Ratio', 'Landmark Coordinates', 'Severity Frequency', 'Shoulder-Ear Distance', 'Angles'))

if st.button("Analyze"):
    if video_file_path:
        with st.spinner('Analyzing video...'):

            result = analyze_video(video_file_path)

            st.write("## Prediction Result")

            if view_option == 'Normal/Hunched Ratio':
                st.write(f"Normal: {result['normal_ratio']}%, Hunched: {result['hunched_ratio']}%")
            elif view_option == 'Landmark Coordinates':
                st.json(result['landmarks_info'])
            elif view_option == 'Severity Frequency':
                st.json(result['status_frequencies'])
            elif view_option == 'Shoulder-Ear Distance':
                distances = [info['vertical_distance_cm'] for info in result['landmarks_info']]
                st.write(f"Distances (cm): {distances}")
            elif view_option == 'Angles':
                angles = [info['angle'] for info in result['landmarks_info']]
                st.write(f"Angles: {angles}")
    else:
        st.error("Please enter a valid video file path.")