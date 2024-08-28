import cv2
import os
import facemesh
import numpy as np

# 동영상 파일 경로
# video_paths = ['../motionMp4/handDates.mp4', '../motionMp4/armscrossedDates.mp4', '../motionMp4/faceDates.mp4','../motionMp4/earDates.mp4',]
# motion = ['hand', 'armscrossed', 'face','ear']


video_paths = [r"C:\Users\user\Desktop\과제, 강의\프로젝트\image\Joy.mp4", r"C:\Users\user\Desktop\과제, 강의\프로젝트\image\talk.mp4", r"C:\Users\user\Desktop\과제, 강의\프로젝트\image\NotJoy.mp4"]
motion = ['Joy', 'talk','NotJoy']





def date_set(xdates, ydates):
    for i, video_path in enumerate(video_paths):
        # 동영상 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        # FPS(초당 프레임 수) 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 0.5초마다 프레임 저장 (프레임 간격 계산)
        frame_interval = int(fps * 0.3)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 동영상 끝
            if frame_count % frame_interval == 0:
                v = facemesh.get_face_mesh_vector(frame)
                xdates.append(v)
                ydates.append(motion[i])
                saved_count += 1
            frame_count += 1
        
        print(f'{motion[i]} : {saved_count}개의 사진을 저장 및 분석하였다.')
        
        # 동영상 캡처 객체 해제
        cap.release()

    cv2.destroyAllWindows()