import cv2
import os
import hyechangMediapipe
import sangwooFace
import numpy as np



# 동영상 파일 경로
video_path = r"C:\tico_project\image\fullmedia.mp4"

# 동영상 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# FPS(초당 프레임 수) 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)

# 0.5초마다 프레임 저장 (프레임 간격 계산)
frame_interval = int(fps * 0.5)

frame_count = 0
saved_count = 0


def date_set(xdates,ydates):
    global frame_count
    global saved_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 동영상 끝

        

        if frame_count % frame_interval == 0:
            v=hyechangMediapipe.get_pose_vector(frame)
            r=sangwooFace.emotion_result(frame)
            print(r)
            xdates.append(v)
            ydates.append(r)
            
#             print('백터 : ')
#             print(v)
#             print('\n감정 결과 : ')
#             print(r)
            saved_count += 1
        frame_count += 1

        

    print(f'{saved_count}개의 사진을 저장 및 분석하였다.')
    # 동영상 캡처 객체 해제
    cap.release()
    cv2.destroyAllWindows()



