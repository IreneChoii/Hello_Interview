import numpy as np
import cv2
from sklearn.svm import SVC
from joblib import load
import hyechangMediapipe

# 사전에 학습된 모델을 단순히 가져와서 사용하는 것
svm_model_what = load('hyechangTraning.joblib')  # 행동이 어떤 행동인지 분석해주는 모델
svm_model_comfort = load('modelTest.joblib')  # 행동이 편한한지 불편한지 분석해주는 모델

motion = 'none'  # 초기값을 문자열로 설정

# 웹캠에서 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 사용합니다.
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 백터화 구하기
    predictVec = hyechangMediapipe.get_pose_vector(frame)
    
    if not predictVec:
        print('모션이 없음')
        continue
    
    predictResult = svm_model_comfort.predict([predictVec])
    print(predictResult[0])
    
    if predictResult[0] == 'uncomfortable':
        motion = svm_model_what.predict([predictVec])[0]
        print(motion)
    else:
        motion = 'none'
    
    # 예측 결과를 프레임에 표시
    cv2.putText(frame, f"motion: {predictResult[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"partition: {motion}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 프레임 표시[0]
    cv2.imshow('Webcam Capture', frame)
    
    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()