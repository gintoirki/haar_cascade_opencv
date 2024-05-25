import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['NgoTruongGiang','DoiXuanDat','DongPhucQuan','NguyenLeChinh']
unknown_label = "Unknown"
euclid_threshold = 100

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, euclid = face_recognizer.predict(faces_roi)

        if euclid < euclid_threshold:
            name = people[label]
        else:
            name = unknown_label

        print(f'Label = {name} with a euclid of {euclid}')

        cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.imshow('Recognnition Face', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()