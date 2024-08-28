import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh

# 입 주변 랜드마크 인덱스
LIPS = [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 146, 178, 179, 180, 181, 182, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 406, 407, 408, 409, 415]

def get_mouth_vector(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        mouth_vector = []
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for idx in LIPS:
                landmark = landmarks[idx]
                mouth_vector.extend([
                    round(landmark.x, 4),
                    round(landmark.y, 4),
                    round(landmark.z, 4)
                ])
        
        return mouth_vector

def analyze_expression(mouth_vector):
    if not mouth_vector:
        return "얼굴을 찾을 수 없음"
    
    # 입의 세로 길이 (높이)
    mouth_height = max([v for i, v in enumerate(mouth_vector) if i % 3 == 1]) - min([v for i, v in enumerate(mouth_vector) if i % 3 == 1])
    
    # 입의 가로 길이 (너비)
    mouth_width = max([v for i, v in enumerate(mouth_vector) if i % 3 == 0]) - min([v for i, v in enumerate(mouth_vector) if i % 3 == 0])
    
    # 입의 면적 (대략적인 계산)
    mouth_area = mouth_height * mouth_width
    
    # 입의 가로-세로 비율
    aspect_ratio = mouth_width / mouth_height if mouth_height > 0 else 0

    # 임계값 설정 (이 값들은 조정이 필요할 수 있습니다)
    JOY_THRESHOLD = 2.5  # 웃을 때의 가로-세로 비율 임계값
    TALK_THRESHOLD = 0.05  # 말할 때의 입 면적 임계값

    if aspect_ratio > JOY_THRESHOLD:
        return "joy"
    elif mouth_area > TALK_THRESHOLD:
        return "talk"
    else:
        return "not joy"

cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    if not ret:
        break
    
    mouth_vector = get_mouth_vector(image)
    expression = analyze_expression(mouth_vector)
    print(f"표정: {expression}")

    cv2.putText(image, f"{expression}", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        
    cv2.imshow('Emotion Detection', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()