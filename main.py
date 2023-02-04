import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('ai_log.xml')

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        print(x, y, w, h)
        roi_gray = gray[y: y + h, x:x + w]  # (y_cord_start, y_cord_end)
        roi_color = frame[y: y + h, x:x + w]

        # recognize? deep learnd model predict keras tensorflow pytorch scikit learn

        # img_item = 'data_face/my_image.png'
        # cv2.imwrite(img_item, roi_gray)

        color = (0, 255, 0)  # BGR  # Green
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('s'):
        break

    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()