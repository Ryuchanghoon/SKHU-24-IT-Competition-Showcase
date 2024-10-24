import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter
import math
import os
import random



model = tf.keras.models.load_model('final_baro_model.h5')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)


def draw_skeleton(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

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

def extract_frames(video_file, interval = 1):
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

def calculate_posture_ratios_based_on_angle(conditions):
    fine_or_normal = ['Fine', 'Danger']
    hunched_conditions = ['Serious', 'Very Serious']

    total_conditions = len(conditions)
    normal_count = sum(1 for cond in conditions if cond in fine_or_normal)
    hunched_count = sum(1 for cond in conditions if cond in hunched_conditions)

    normal_ratio = (normal_count / total_conditions) * 100
    hunched_ratio = (hunched_count / total_conditions) * 100

    return hunched_ratio, normal_ratio

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

def record_video(output_path, duration=5):
    cap = cv2.VideoCapture(1)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    st.write("영상 녹화 중~ 좀 걸려요")

    captured_frames = []  
    start_time = cv2.getTickCount()
    while int((cv2.getTickCount() - start_time) / cv2.getTickFrequency()) < duration:
        ret, frame = cap.read()
        if ret:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        
            if results.pose_landmarks:
                draw_skeleton(frame, results)

            out.write(frame)
            captured_frames.append(frame)
        else:
            break

    cap.release()
    out.release()

    if captured_frames:
        random_frame = random.choice(captured_frames) 
        st.image(random_frame, channels="BGR")  
    st.write("분석 완료")


st.title("BARO")
st.write("## 왼쪽 귀랑 어깨가 잘 보이게 앉아 주세요" )


view_option = st.radio("Choose result:", ('Normal/Hunched Ratio', 'Landmark Coordinates', 'Severity Frequency', 'Shoulder-Ear Distance', 'Angles'))

if st.button("Start Recording and Analyze"):
    video_output_path = "recorded_vid.avi"
    with st.spinner('Recording... 좀 기다려 주세요~'):
        record_video(video_output_path, duration = 5)

    with st.spinner('분석 중... 좀 걸려요'):
        result = analyze_video(video_output_path)

        st.write("## 자세 분석 결과는?????????")

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
