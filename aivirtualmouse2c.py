import mediapipe as mp
import numpy as np
import cv2
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 미디어파이프로 손 인식 모델 초기화
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 화면 크기 구하기
screen_size = pyautogui.size()

# 마우스 이동 범위 설정 (화면 크기 내에서)
move_range = np.array([screen_size[0], screen_size[1]]) * 0.8

# 손가락 인덱스 (0부터 시작)
INDEX_FINGER = 8
MIDDLE_FINGER = 12

# 마우스 이동 함수
def move_mouse(x, y):
    x = int(x * move_range[0])
    y = int(y * move_range[1])
    pyautogui.moveTo(x, y)

# 마우스 클릭 함수
def click_mouse():
    pyautogui.click()

# 키보드 입력 함수
def press_key(key):
    pyautogui.press(key)

# 카메라 캡처 초기화
cap = cv2.VideoCapture(0)

def run_app():
    while True:
        success, image = cap.read()
        if not success:
            break

        # 이미지를 BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 미디어파이프로 손 인식
        results = hands.process(image)

        if results.multi_hand_landmarks:
            # 이미지에 손가락 좌표 그리기
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손가락 좌표 추출
                index_finger = hand_landmarks.landmark[INDEX_FINGER]
                middle_finger = hand_landmarks.landmark[MIDDLE_FINGER]

                # 마우스 이동
                move_mouse(index_finger.x, index_finger.y)

                # 마우스 클릭
                if middle_finger.y < index_finger.y:
                    click_mouse()

                # 키보드 입력
                if middle_finger.y > index_finger.y:
                    press_key('space')

        # 이미지 출력
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_app()