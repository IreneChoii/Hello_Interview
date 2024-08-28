import numpy as np
import cv2
from sklearn.svm import SVC
import mp4Analysis as mn
import hyechangMediapipe 
from joblib import dump

#행동이 편한지 불편한지 분석해주는 모델

xdates=[]
ydates=[]

mn.date_set(xdates,ydates)
svm_model=SVC(kernel='linear', C=1, random_state=42)



# my_xdates=np.array(xdates)
# my_ydates=np.array(ydates)


# a=np.array(xdates)
# print(a.ndim)

print('import 완료')

svm_model.fit(xdates,ydates)
print('사전 학습 완료 modelTest.py')

#모델 저장
dump(svm_model,'modelTest.joblib')
print('modelTest.joblib 모델 저장')

