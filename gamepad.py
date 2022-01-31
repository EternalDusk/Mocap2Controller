import cv2
import numpy as np
import mouseHandTrackingModule as htm
import time
import autopy
import pyxinput

##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 3
#########################

virtualController = pyxinput.vController()

virtualRead = pyxinput.rController(1)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(3)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

clicked = False

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
    
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
        (255, 0, 255), 2)
        # 4. Only Index Finger : Moving Mode
        
        #if thumb is up, control is left joystick, or movement
        #else, control is right joystick, or looking direction
        if fingers[4] == 1:
            virtualController.set_value('AxisRx', 0)
            virtualController.set_value('AxisRy', 0)
            if fingers[1] == 1 and fingers[2] == 0:
            
                
                # 5. Convert Coordinates
                xconvCord = -1 * (320 - x1)
                yconvCord = (240 - y1)
                
                xStick = xconvCord/100
                yStick = yconvCord/100
                
                if xStick > 1:
                    xStick = 1
                elif xStick < -1:
                    xStick = -1
                
                if yStick > 1:
                    yStick = 1
                elif yStick < -1:
                    yStick = -1
                
                # 6. Smoothen Values
                #clocX = plocX + (x3 - plocX) / smoothening
                #clocY = plocY + (y3 - plocY) / smoothening
            
                # 7. Move Mouse
                cv2.circle(img, (320, 240), 100, (0, 0, 255), 3)
                cv2.line(img, (x1, y1), (320, 240), (0, 0, 255), 3)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                
                virtualController.set_value('AxisLx', xStick)
                virtualController.set_value('AxisLy', yStick)
                print(virtualRead.gamepad)
                virtualController.set_value('BtnA', 0)
                
            # 8. Both Index and middle fingers are up : Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # 9. Find distance between fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)
                print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                    15, (0, 255, 0), cv2.FILLED)
                    virtualController.set_value('BtnA', 255)
        else:
            virtualController.set_value('AxisLx', 0)
            virtualController.set_value('AxisLy', 0)
            if fingers[1] == 1 and fingers[2] == 0:
            
                
                # 5. Convert Coordinates
                xconvCord = -1 * (320 - x1)
                yconvCord = (240 - y1)
                
                xStick = xconvCord/100
                yStick = yconvCord/100
                
                if xStick > 1:
                    xStick = 1
                elif xStick < -1:
                    xStick = -1
                
                if yStick > 1:
                    yStick = 1
                elif yStick < -1:
                    yStick = -1
                
                # 6. Smoothen Values
                #clocX = plocX + (x3 - plocX) / smoothening
                #clocY = plocY + (y3 - plocY) / smoothening
            
                # 7. Move Mouse
                cv2.circle(img, (320, 240), 100, (0, 0, 255), 3)
                cv2.line(img, (x1, y1), (320, 240), (0, 0, 255), 3)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                
                virtualController.set_value('AxisRx', xStick)
                virtualController.set_value('AxisRy', yStick)
                print(virtualRead.gamepad)
                virtualController.set_value('TriggerR', 0)
                
            # 8. Both Index and middle fingers are up : Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # 9. Find distance between fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)
                print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                    15, (0, 255, 0), cv2.FILLED)
                    virtualController.set_value('TriggerR', 255)
    
    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)