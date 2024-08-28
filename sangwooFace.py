import cv2
import time
from google.cloud import vision
import numpy as np
import os

# Google Cloud 인증 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\tico_project\propane-girder-429404-g6-e6b600c1a3e6.json"

# Vision API 클라이언트 초기화
client = vision.ImageAnnotatorClient()

def detect_face_and_emotion(image):
    _, encoded_image = cv2.imencode('.jpg', image)
    image_content = encoded_image.tobytes()
    image = vision.Image(content=image_content)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    if faces:
        face = faces[0]  # 첫 번째 감지된 얼굴 사용
        emotions = {
            'Joy': face.joy_likelihood,
            'Sorrow': face.sorrow_likelihood,
            'Anger': face.anger_likelihood,
            'Surprise': face.surprise_likelihood
        }
        return emotions, face.bounding_poly.vertices
    return None, None

def likelihood_to_percentage(likelihood):
    mapping = {
        vision.Likelihood.UNKNOWN: 0,
        vision.Likelihood.VERY_UNLIKELY: 0,
        vision.Likelihood.UNLIKELY: 25,
        vision.Likelihood.POSSIBLE: 50,
        vision.Likelihood.LIKELY: 75,
        vision.Likelihood.VERY_LIKELY: 100}
     
    return mapping.get(likelihood, 5)

def emotion_result(image):
    emotions, face_vertices = detect_face_and_emotion(image)
    
    if emotions and face_vertices:
        emotion_percentages = {emotion: likelihood_to_percentage(likelihood)
                               for emotion, likelihood in emotions.items()}
        
        face_detected = True
        print('face detected')
        
        print(likelihood_to_percentage(emotions['Surprise']))
        print(likelihood_to_percentage(emotions['Joy']))
        
        
        max_emotion = max(emotion_percentages, key=emotion_percentages.get)
        max_percentage = emotion_percentages[max_emotion]
        print('max emotion : ',max_emotion)

        cv2.putText(image, f"{likelihood_to_percentage(emotions['Surprise']), likelihood_to_percentage(emotions['Joy'])}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)   
        cv2.imshow('test',image)
# 키 입력 대기 (1ms)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return 'quit'  # 'q'가 눌렸을 때 'quit' 반환
        
        if likelihood_to_percentage(emotions['Surprise']) == likelihood_to_percentage(emotions['Joy']):
            return 'uncomfortable'
        elif max_percentage < 21:
            return 'comfortable'
        elif max_emotion in ['Sorrow', 'Anger', 'Surprise']:
            return 'uncomfortable'
        elif max_emotion == 'Joy':
            return 'comfortable'
    else:
        print('no face detected')
        return '없음'
    

def joyTest() :
     cap = cv2.VideoCapture(0)
     while True:
        ret, image = cap.read()
        if not ret:
            break
        emotions, face_vertices = detect_face_and_emotion(image)
        
        if emotions and face_vertices:
            emotion_percentages = {emotion: likelihood_to_percentage(likelihood)
                                for emotion, likelihood in emotions.items()}

            face_detected = True
            print('face detected')
            
            
            max_emotion = max(emotion_percentages, key=emotion_percentages.get)
            max_percentage = emotion_percentages[max_emotion]


            # 키 입력 대기 (1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return 'quit'  # 'q'가 눌렸을 때 'quit' 반환
            
            result=None

            if max_percentage < 24:
                result= 'notJoy'
            elif max_emotion == 'Joy' and max_percentage >= 50:
                result= 'Joy'
            else:
                result= 'notJoy'
            
            cv2.putText(image, f"{result}", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
            y_offset = 30
            for emotion, likelihood in emotions.items():
                percentage = likelihood_to_percentage(likelihood)
                cv2.putText(image, f"{emotion}: {percentage}%", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30    

            cv2.imshow('test',image)
        else:
            print('no face detected')
            return '없음'

