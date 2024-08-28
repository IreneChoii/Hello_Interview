import numpy as np
import cv2
from sklearn.svm import SVC
import mp4MotionAnalysis as ma
import hyechangMediapipe 
from joblib import dump

print('모듈 import 완료')

# 데이터 준비 및 모델 학습
xdates = []
ydates = []
ma.date_set(xdates, ydates)

# xdates의 구조 확인
for i, x in enumerate(xdates):
    print(f"xdates[{i}] shape: {np.array(x).shape}")

# 데이터 전처리 (예시)
max_len = max(len(x) for x in xdates)
xdates_padded = [x + [0]*(max_len - len(x)) for x in xdates]

X = np.array(xdates_padded)
y = np.array(ydates)

print(X.ndim)

svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X, y)
print('모델 학습 완료')

# 모델 저장
dump(svm_model, 'hyechangTraning.joblib')
print('hyechangTraning 모델 저장')