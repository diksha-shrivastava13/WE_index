import cv2
import numpy as np
import time
import autopy
import track_hand

wCam, hCam = 640, 480                       # width and height of cam screen
cap = cv2.VideoCapture(0)                   # set the parameter to be the index of your webcam
cap.set(3, wCam)                            # 3 is for width, 4 is for height, pre-defined indexes
cap.set(4, hCam)
pTime = 0
wScr, hScr = autopy.screen.size()
frameR = 100                                # Frame Reduction
smoothening = 7                             # random value
detector = track_hand.HandDetector(maxhands=1)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            cv2.circle(img, (x1, y1), 20, (0, 0, 205), cv2.FILLED)

            # Add code here to redirect to video
            # print("Hola Asokan!")

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 80), cv2.FONT_ITALIC, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == 13:
        cv2.destroyAllWindows()
        break

