import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh

def get_face_mesh_vector(image):
    """
    이미지에서 얼굴 메시를 추정하고, 얼굴 랜드마크를 벡터화하여 반환합니다.
    매개변수:
    - image: 입력 이미지 (numpy 배열)
    반환값:
    - face_vector: 얼굴 메시 랜드마크 벡터 (리스트)
    """
    # MediaPipe Face Mesh 모델 초기화
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 메시 추정 결과를 얻음
        results = face_mesh.process(image_rgb)
        
        # 얼굴 랜드마크를 벡터화하기 위한 리스트
        face_vector = []
        
        # 얼굴 랜드마크가 존재하는 경우
        if results.multi_face_landmarks:
            for landmark in results.multi_face_landmarks[0].landmark:
                face_vector.extend([
                    round(landmark.x, 4),
                    round(landmark.y, 4),
                    round(landmark.z, 4)
                ])
        
        return face_vector
