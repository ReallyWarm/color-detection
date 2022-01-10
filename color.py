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

                # Get name
                color = cv2.getTrackbarPos("Color", "Controls")
                name = "Red" if color == 0 else "Orange" if color == 1 else "Yellow" if color == 2 else "Green" \
                        if color == 3 else "Blue" if color == 4 else "Purple"
                cv2.putText(frame, name, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)

# Color range
lower_red = np.array([158, 105, 84])
upper_red = np.array([180, 255, 255])

lower_orange = np.array([7, 115, 140])
upper_orange = np.array([22, 255, 255])

lower_yellow = np.array([23, 86, 80])
upper_yellow = np.array([39, 255, 255])

lower_green = np.array([40, 86, 80])
upper_green = np.array([83, 255, 255])

lower_blue = np.array([84, 86, 80], dtype=np.uint8)
upper_blue = np.array([126, 255, 255], dtype=np.uint8)

lower_purple = np.array([127, 86, 80])
upper_purple = np.array([157, 255, 255])


# Trackbar window
control_win = "Controls"
cv2.namedWindow(control_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(control_win, 640, 400)
# All trackbar
cv2.createTrackbar("Area", control_win, 200, 20000, empty)
cv2.createTrackbar("Color", control_win, 0, 5, empty)

while 1:
    ret, frame = cap.read()
    blur_frame = cv2.GaussianBlur(frame, (7, 7), 1)
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # Select color
    color = cv2.getTrackbarPos("Color", "Controls")
    if color == 0:
        mask = cv2.inRange(hsv, lower_red, upper_red)
    elif color == 1:
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
    elif color == 2:
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    elif color == 3:
        mask = cv2.inRange(hsv, lower_green, upper_green)
    elif color == 4:
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    else:
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Mask
    bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    object = cv2.bitwise_and(frame, frame, mask=mask)
    frame_cnts = frame.copy()

    # Draw contours and rectangle
    drawContours(mask, object) 
    drawContours(mask, frame_cnts, cnts=False, rect=True)

    # Resize
    frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    frame_cnts = cv2.resize(frame_cnts, (0, 0), None, 0.5, 0.5)
    bgr_mask = cv2.resize(bgr_mask, (0, 0), None, 0.5, 0.5)
    object = cv2.resize(object, (0, 0), None, 0.5, 0.5)

    row1 = np.hstack((frame, frame_cnts)) # show video and color rects
    row2 = np.hstack((bgr_mask, object)) # show B&W mask and object with contours
    result = np.vstack((row1, row2))

    cv2.imshow("frame", result) # show video

    if cv2.waitKey(30) == ord("z"):
        break

cap.release()
cv2.destroyAllWindows()