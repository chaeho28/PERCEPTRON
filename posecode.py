import cv2
import mediapipe as mp
import numpy as np
import time

class PoseTracking():
    def __init__(self, mode = False, complexity = 1, landmarks = True, min_detection_confidence = 0.6, min_tracking_confidence = 0.6):
        self.mode = mode
        self.complexity = complexity
        self.landmarks = landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode = self.mode,
            model_complexity = self.complexity,
            smooth_landmarks = self.landmarks,
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence = self.min_tracking_confidence
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findpose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
              if draw:
                   self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        return img
    

def main():
    cap = cv2.VideoCapture(0)
    pt = PoseTracking()
    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            continue
        img = pt.findpose(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()