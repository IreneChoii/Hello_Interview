#최종 백터화!!!


import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose





def get_pose_vector(image):
    """
    이미지에서 포즈를 추정하고, 포즈 랜드마크를 벡터화하여 반환합니다.

    매개변수:
    - image: 입력 이미지 (numpy 배열)

    반환값:
    - pose_vector: 포즈 랜드마크 벡터 (리스트)
    """
    # Mediapipe 포즈 추정기를 초기화합니다
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        # BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 포즈 추정 결과를 얻음
        results = pose.process(image_rgb)
        
        # 포즈 랜드마크를 벡터화하기 위한 리스트
        pose_vector = []

        # 포즈 랜드마크가 존재하는 경우
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose_vector.extend([
            round(landmark.x, 4),
            round(landmark.y, 4),
            round(landmark.z, 4),
            round(landmark.visibility, 4)
        ])

        
        return pose_vector





