# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response, request,jsonify,send_file
import cv2
import os
import speech_recognition as sr
import threading
from pydub import AudioSegment
import myGPT as mg
import time
import swMotionFeedback
import hyechangMediapipe
from joblib import load

# 사전에 학습된 모델을 단순히 가져와서 사용하는 것
svm_model_what = load('hyechangTraning.joblib')  # 행동이 어떤 행동인지 분석해주는 모델
svm_model_comfort = load('modelTest.joblib')  # 행동이 편한한지 불편한지 분석해주는 모델


app = Flask(__name__)

recognizer = sr.Recognizer()
audio_segments = []  # 여기에 오디오 세그먼트를 저장합니다
recording_thread = None

# 질문과 대답
count = -1
q_count=-1
answer = ''
question = ''
questions = []
answers = []

# OpenCV로 얼굴 촬영을 위한 변수 설정
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
output_folder = 'recordings'  # 저장할 폴더 설정
os.makedirs(output_folder, exist_ok=True)  # 폴더가 없으면 생성

is_recording = False
out = None  # VideoWriter 객체를 전역 변수로 설정

fps = 12.0  # 사용할 프레임 속도를 고정

def capture_face():
    global is_recording
    global output_folder
    global out
    global fps
    
    start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_count = 0
    
    while is_recording:
        ret, frame = camera.read()
        if ret:
            if out is None:
                file_name = os.path.join(output_folder, 'hyechangTest.mp4')
                out = cv2.VideoWriter(file_name, fourcc, fps, (frame.shape[1], frame.shape[0]))

            out.write(frame)
            frame_count += 1

            # 프레임 레이트 조절
            elapsed_time = time.time() - start_time
            expected_frame_count = int(elapsed_time * fps)
            if frame_count > expected_frame_count:
                time.sleep((frame_count - expected_frame_count) / fps)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 영상 저장 완료 후 리소스 해제
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    print(f"Total frames: {frame_count}, Total time: {elapsed_time:.2f} seconds")

def generate_frames():
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def generate_frames_test():
    global camera
    while True:
        success, frame = camera.read()
        
        if not success:
            break
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

        
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def record_audio():
    global audio_segments
    with sr.Microphone() as source:
        while True:  # 무한히 녹음을 수행합니다
            print("Listening...")
            audio_data = recognizer.listen(source, phrase_time_limit=5)  # 5초 제한으로 오디오를 받습니다
            audio_segments.append(audio_data)  # 받은 오디오를 세그먼트에 추가합니다
            # recording_thread이 None이 되면 스레드 종료
            if recording_thread is None:
                break

@app.route('/')
def index():
    return render_template('information.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    print("interview.html 실행")
    return render_template('interview.html')

@app.route('/cameraTest', methods=['POST'])
def cameraTest():
    print("cameraTest.html 실행")
    return render_template('cameraTest.html')

@app.route('/show_result1', methods=['POST'])
def show_result1():
    print("showResult1.html 실행")
    return render_template('showResult1.html')

@app.route('/show_result2', methods=['POST'])
def show_result2():
    print("showResult2.html 실행")
    return render_template('showResult2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_best_img')
def get_best_img():
    return send_file('image/best.jpg', mimetype='image/jpeg')

@app.route('/video_check_test')
def video_check_test():
    return Response(generate_frames_test(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analysis-Video', methods=['POST'])
def analysis_Video():
    # 상우가 만든 motionFeedback()
    mp4Analysis,final_comfort_ratio,final_uncomfort_ratio,final_hand_ratio,final_face_ratio,final_head_ratio,final_arm_ratio,final_shoulder_ratio,final_handtouch_ratio=swMotionFeedback.motionFeedback()
    
    return jsonify({
        "mp4Analysis": mp4Analysis,
        "final_comfort_ratio": final_comfort_ratio,
        "final_uncomfort_ratio": final_uncomfort_ratio,
        "final_hand_ratio": final_hand_ratio,
        "final_face_ratio": final_face_ratio,
        "final_head_ratio": final_head_ratio,
        "final_arm_ratio": final_arm_ratio,
        "final_shoulder_ratio": final_shoulder_ratio,
        "final_handtouch_ratio": final_handtouch_ratio    
    })



@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording
    global out
    global camera
    global fps
    
    if not is_recording:
        is_recording = True
        
        # 카메라 설정 확인 및 조정
        camera.set(cv2.CAP_PROP_FPS, fps)  # FPS를 설정
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        print(f"Camera FPS: {actual_fps}")
        
        threading.Thread(target=capture_face).start()
        print('녹화시작')
        return 'Recording started!'
    else:
        print('이미 녹화중입니다.')
        return 'Already recording!'

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    global out
    global fps
    
    if is_recording:
        is_recording = False
        time.sleep(1/fps) 
        if out is not None:
            out.release()
            out = None
        print('녹화종료')
        
        return 'Recording stopped!'
    else:
        return 'Not recording!'

@app.route('/create-question', methods=['POST'])
def create_question():
    global questions
    print('count check')
    count = request.args.get('count', default=1, type=int)
    print(count)
    questions=mg.first_question(count)

    return '질문 생성 성공'

@app.route('/get_question', methods=['POST'])
def get_question():
    global questions
    global q_count
    print("get_question 호출")
    q_count+=1
    print(questions[q_count])
    return questions[q_count]
    

@app.route('/analysis-answer', methods=['POST'])
def analysis_answer():
    global count
    currentCount = request.args.get('currentCount', default=1, type=int)
    print('analysis_answer check test!')
    
    print(f'현재 {currentCount} 질문 : {questions[currentCount]} 대답 : {answers[currentCount]}')
    return mg.analysis_answer(questions[currentCount], answers[currentCount])

@app.route('/start_audio_recording', methods=['POST'])
def start_audio_recording():
    global recording_thread
    global audio_segments
    if recording_thread is None or not recording_thread.is_alive():
        # 녹음 스레드가 없거나 종료된 경우 새로 시작합니다
        recording_thread = threading.Thread(target=record_audio)
        audio_segments = []  # 오디오 세그먼트 초기화
        recording_thread.start()
        return "녹음이 시작되었습니다."
    else:
        return "이미 녹음 중입니다."

@app.route('/stop_audio_recording', methods=['POST'])
def stop_audio_recording():
    global recording_thread
    global audio_segments
    global answer
    print('녹음중지 시도중')
    try:
        # recording_thread를 중지시키고, 오디오 세그먼트를 WAV 파일로 저장하고, 이를 합쳐서 combined.wav 파일을 생성합니다
        recording_thread = None
        combined_audio = AudioSegment.empty()
        for i, audio_data in enumerate(audio_segments):
            with open(f"segment_{i}.wav", "wb") as f:
                f.write(audio_data.get_wav_data())

            audio = AudioSegment.from_wav(f"segment_{i}.wav")
            combined_audio += audio

        combined_audio.export("combined.wav", format="wav")

        # 합쳐진 오디오 파일로 인식을 수행합니다
        with sr.AudioFile("combined.wav") as source:
            audio = recognizer.record(source)

        # 인식된 텍스트를 반환합니다
        text = recognizer.recognize_google(audio, language='ko-KR')
        answer = text
        answers.append(text)
        return text
    except sr.UnknownValueError:
        answers.append("음성을 인식할 수 없습니다.")
        return "음성을 인식할 수 없습니다."
    except sr.RequestError as e:
        answers.append("음성을 인식할 때 에러가 발생했습니다.")
        return "음성을 인식할 때 에러가 발생했습니다."

if __name__ == '__main__':
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        print("서버 시작 시도...")
    app.run(debug=True)
