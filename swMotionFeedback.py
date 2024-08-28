import numpy as np
import cv2
from sklearn.svm import SVC
from joblib import load
import hyechangMediapipe
import os
import time
import myGPT


#전역변수 비율
final_comfort_ratio=0
final_uncomfort_ratio=0
final_hand_ratio=0
final_face_ratio=0
final_head_ratio=0
final_arm_ratio=0
final_shoulder_ratio = 0
final_handtouch_ratio = 0


def motionFeedback():
    global final_comfort_ratio,final_uncomfort_ratio,final_hand_ratio,final_face_ratio,final_head_ratio,final_arm_ratio,final_shoulder_ratio,final_handtouch_ratio


    # 사전에 학습된 모델을 가져옵니다
    svm_model_what = load('hyechangTraning.joblib')
    svm_model_comfort = load('modelTest.joblib')

    # 동영상 파일 경로
    video_path = r"C:\tico_project\recordings\hyechangTest.mp4"
    if os.path.exists(video_path)==False:
        time.sleep(3)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("동영상 파일을 열 수 없습니다.")
        exit()

    # 편한/불편한 상태 카운터 초기화


    fps = cap.get(cv2.CAP_PROP_FPS)

    # 0.5초마다 프레임 저장 (프레임 간격 계산)
    frame_interval = int(fps * 1)

    frame_count = 0
    saved_count = 0
    comfort_count = 0
    uncomfort_count = 0
    total_frames = 0

    handRatio = 0
    faceRatio = 0
    headRatio = 0
    armRatio = 0
    shoulderRatio = 0
    handtouchRatio = 0
    
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            total_frames += 1
        
            predictVec = hyechangMediapipe.get_pose_vector(frame)
            if not predictVec:
                print('모션이 없음')
                continue
        
            predictResult = svm_model_comfort.predict([predictVec])

            print(predictResult[0])
            if predictResult[0] == 'comfortable':
                comfort_count += 1
                
            else:  # 'uncomfortable'
                uncomfort_count += 1
                motion = svm_model_what.predict([predictVec])[0]
                if motion == 'hand' :
                    handRatio += 1
                elif motion == 'head' :
                    headRatio += 1
                elif motion == 'face' :
                    faceRatio += 1
                elif motion == 'shoulder' :
                    shoulderRatio += 1
                elif motion == 'handtouch' :
                    handtouchRatio += 1
                else :
                    armRatio += 1

                print(motion)

        frame_count += 1

    # 비디오 캡처 객체 해제
    cap.release()

    # 결과 계산 및 출력
    total_analyzed_frames = comfort_count + uncomfort_count

    print('total_analyzed_frames : ',total_analyzed_frames)
    
    comfort_ratio = comfort_count / total_analyzed_frames * 100
    uncomfort_ratio = uncomfort_count / total_analyzed_frames * 100

    total_motions = handRatio + headRatio + faceRatio + armRatio + shoulderRatio + handtouchRatio

    hand_result = (handRatio / total_motions * 100) if total_motions > 0 else 0
    head_result = (headRatio / total_motions * 100) if total_motions > 0 else 0
    face_result = (faceRatio / total_motions * 100) if total_motions > 0 else 0
    arm_result = (armRatio / total_motions * 100) if total_motions > 0 else 0
    shoulder_result = (shoulderRatio / total_motions * 100) if total_motions > 0 else 0
    handtouch_result = (handtouchRatio / total_motions * 100) if total_motions > 0 else 0
    
    


    final_comfort_ratio=comfort_ratio
    final_uncomfort_ratio=uncomfort_ratio
    final_hand_ratio=hand_result
    final_face_ratio=face_result
    final_head_ratio=head_result
    final_arm_ratio=arm_result
    final_shoulder_ratio=shoulder_result
    final_handtouch_ratio=handtouch_result



    

    
    print(f"\n분석 결과:")
    print(f"총 프레임 수: {total_frames}")
    print(f"분석된 프레임 수: {total_analyzed_frames}")
    print(f"편한 상태 비율: {comfort_ratio:.2f}%")
    print(f"불편한 상태 비율: {uncomfort_ratio:.2f}%")

    print(f"hand 상태 비율: {hand_result:.2f}%")
    print(f"head 상태 비율: {head_result:.2f}%")
    print(f"face 상태 비율: {face_result:.2f}%")
    print(f"arm 상태 비율: {arm_result:.2f}%")
    print(f"shoulder 상태 비율: {shoulder_result:.2f}%")
    print(f"handtouch 상태 비율: {handtouch_result:.2f}%")
    
    
    return myGPT.analysis_video(comfort_ratio, uncomfort_ratio,hand_result,head_result,face_result,arm_result,handtouch_result,shoulder_result),final_comfort_ratio,final_uncomfort_ratio,final_hand_ratio,final_face_ratio,final_head_ratio,final_arm_ratio,final_shoulder_ratio,final_handtouch_ratio


