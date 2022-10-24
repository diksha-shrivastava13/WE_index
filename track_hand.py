import cv2
import mediapipe
import math


class HandDetector:
    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.trackcon = trackcon
        self.detectioncon = detectioncon
        self.maxhands = maxhands
        self.mphands = mediapipe.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxhands, self.detectioncon, self.trackcon)
        self.mpdraw = mediapipe.solutions.drawing_utils
        self.tipids = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handno=0, draw=True):
        xList, yList, bbox = [], [], []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for ide, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([ide, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipids[0]][1] > self.lmList[self.tipids[0] -1][1]:
            fingers.append(0)
        else:
            fingers.append(1)
        for ele in range(1, 5):
            if self.lmList[self.tipids[ele]][2] > self.lmList[self.tipids[ele] - 2][2]:
                fingers.append(0)
            else:
                fingers.append(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2)//2, (y1+y2)//2
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, x2, y1, y2, cx, cy]

