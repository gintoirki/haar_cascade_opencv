import os
import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

video_capture = cv.VideoCapture(0)

output_folder = r"D:\python_code\pythonProject\Coursera\XuLyAnh\crop"
os.makedirs(output_folder, exist_ok=True)

counter = 0

while True:

    ret, frame = video_capture.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        cv.imwrite(os.path.join(output_folder, f"face_{counter}.jpg"), face_image)
        counter += 1
        cv.imshow("Face", face_image)

    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2destroyAllWindows()