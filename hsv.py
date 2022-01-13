import cv2
import numpy as np

def empty(x):
    pass

def drawContours(mask, frame, cnts=True, rect=False):
    if not cnts and not rect:
        return

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        areaMin = cv2.getTrackbarPos("Area", "Controls")
        if cv2.contourArea(cnt) > areaMin:
            if cnts:
                cv2.drawContours(frame, cnt, -1, (255, 0, 255), 2)
            
            # Draw rectangle
            if rect:
                length = cv2.arcLength(cnt, True)
                shape = cv2.approxPolyDP(cnt, 0.02 * length, True)
                x, y, w, h = cv2.boundingRect(shape)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Draw center
                centerX = x + (w//2)
                centerY = y + (h//2)
                cv2.circle(frame, (centerX, centerY), 4, (0, 0, 0), -1)
                cv2.putText(frame, "center", (centerX - 22 , centerY - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 20, 20), 2)


cap = cv2.VideoCapture(0)

# Trackbar window
control_win = "Controls"
cv2.namedWindow(control_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(control_win, 640, 400)
# All trackbar
cv2.createTrackbar("Area", control_win, 200, 20000, empty)
cv2.createTrackbar("HUE_Low", control_win, 0, 180, empty)
cv2.createTrackbar("SAT_Low", control_win, 100, 255, empty)
cv2.createTrackbar("VAL_Low", control_win, 100, 255, empty)
cv2.createTrackbar("HUE_High", control_win, 50, 180, empty)
cv2.createTrackbar("SAT_High", control_win, 255, 255, empty)
cv2.createTrackbar("VAL_High", control_win, 255, 255, empty)

while 1:
    ret, frame = cap.read()
    blur_frame = cv2.GaussianBlur(frame, (7, 7), 1)
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # Hue Satuation Value
    hue1 = cv2.getTrackbarPos("HUE_Low", control_win)
    hue2 = cv2.getTrackbarPos("HUE_High", control_win)
    sat1 = cv2.getTrackbarPos("SAT_Low", control_win)
    sat2 = cv2.getTrackbarPos("SAT_High", control_win)
    val1 = cv2.getTrackbarPos("VAL_Low", control_win)
    val2 = cv2.getTrackbarPos("VAL_High", control_win)

    hsv_low = np.array([hue1, sat1, val1])
    hsv_high = np.array([hue2, sat2, val2])

    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    result = frame.copy()

    # Draw contours and rectangle 
    drawContours(mask, result, rect=True)

    cv2.imshow("frame", result) # show video

    if cv2.waitKey(30) == ord("z"):
        break

cap.release()
cv2.destroyAllWindows()