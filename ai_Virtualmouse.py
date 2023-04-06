import cv2
import mediapipe as mp
import pyautogui
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands() #미디어 파이프의 손 탐지와 관련된 것
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0
while True:
    _, frame = cap.read() #무한 반복문으로, opencv 라이브러리 공간에서 변수 선언_무한반복
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                if id == 8: #검지의 끝을 의미
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    index_x = screen_width/frame_width*x
                    index_y = screen_height/frame_height*y

                if id == 4: #4번은 엄지의 끝
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                    print('엄지값_외곽', abs(index_y - thumb_y))

                    if abs(index_y - thumb_y) < 20: #이부분을 고치면 자동으로 이동 그리고 변형을 주면, 엔터키 까지 개발 가능
                        pyautogui.click()
                        pyautogui.sleep(1)
                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y)


    cv2.imshow('PERCEPTRON ver2', frame)
    cv2.waitKey(1)