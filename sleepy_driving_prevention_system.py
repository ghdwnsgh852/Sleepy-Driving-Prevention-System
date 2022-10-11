import cv2
import numpy as np
import dlib
import os
import threading
from imutils import face_utils

# i는 음악파일을 on off 하기 위한 변수
# global i
# i=False

# def siren():
#     global i
#     while True:
#         if i:
#             os.system('mpg123 /home/ghdwnsgh852/sss.mp3')
#             i=False

# thead=threading.Thread(target=siren)
# thead.start()

#카메라에서 영상을 받아옴
cap = cv2.VideoCapture(0)

# 얼굴 객체 탐지
detector = dlib.get_frontal_face_detector()
#얼굴의 특징점 탐지
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

warning = 0
active = 0
status=""
color=(0,0,0)

# 길이 구하는 함수
def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist
# 눈이 감겼는지 확인하는 함수
def blinked(a,b,c,d,e,f):
    #눈의 가로 길이 p2 p6의 길이 p3 p5의 길이 측정
    vertical = compute(b,d) + compute(c,e)
    # 눈의 가로 길이 p1 p4의 길이
    horizon = compute(a,f)
    # 가로 세로 비율 측정
    ratio = vertical/(2.0*horizon)
    
    if(ratio>0.25): #임계치는 0.25로 설정
        #눈 뜸
        return 1
        
    else:
        #눈 감음
        return 0 
        

while True:
    #이미지 프레임 가져옴
    _, face_frame = cap.read()
    #이미지에서 얼굴을 찾음
    faces = detector(face_frame)
    #이미지에 사람 없음
    if not faces:
        status='Absence'
        color = (255,0,0)

    for face in faces :
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #얼굴에 사각형 이미지 그림
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(face_frame, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36],landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if(left_blink==0 or right_blink==0): #왼쪽 또는 오른쪽 눈이 감김
            warning+=1
            active=0
            if(warning>6): # 6프레임이상 눈이 감기면 졸고 있다고 판단
                status="WARNING"
                color = (0,0,255)
                # i=True

        else:
            warning=0
            active+=1
            if(active>6):
                status="OK!!"
                color = (0,255,0)
        
        cv2.putText(face_frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)
        #이미지에 status 글자 넣기
        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
            #얼굴에 특징점 그리기

    cv2.imshow("Result of detector", face_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
