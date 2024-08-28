import cv2
import os
import hyechangMediapipe
import numpy as np

# 동영상 파일 경로
# video_paths = ['../motionMp4/handDates.mp4', '../motionMp4/armscrossedDates.mp4', '../motionMp4/faceDates.mp4','../motionMp4/earDates.mp4',]
# motion = ['hand', 'armscrossed', 'face','ear']


video_paths = [r"C:\tico_project\image\handDatas.mp4", r"C:\tico_project\image\armscrossedDatas.mp4", r"C:\tico_project\image\faceDatas.mp4", 
               r"C:\tico_project\image\headDatas.mp4", r"C:\tico_project\image\shoulderDatas.mp4", r"C:\tico_project\image\handtouchDatas.mp4"]
motion = ['hand', 'armscrossed', 'face', 'head', 'shoulder', 'handtouch']





def date_set(xdates, ydates):
    for i, video_path in enumerate(video_paths):
        # 동영상 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        # FPS(초당 프레임 수) 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 0.5초마다 프레임 저장 (프레임 간격 계산)
        frame_interval = int(fps * 0.05)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 동영상 끝
           
            cv2.imshow('headTest',frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return 'quit'  # 'q'가 눌렸을 때 'quit' 반환
        

            
            if frame_count % frame_interval == 0:
                v = hyechangMediapipe.get_pose_vector(frame)
                xdates.append(v)
                ydates.append(motion[i])
                saved_count += 1
            frame_count += 1
        
        print(f'{motion[i]} : {saved_count}개의 사진을 저장 및 분석하였다.')
        
        # 동영상 캡처 객체 해제
        cap.release()

    cv2.destroyAllWindows()