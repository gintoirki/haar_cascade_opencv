import cv2 as cv
import os

# Đường dẫn đến tập tin Haar Cascade XML
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Thư mục chứa ảnh đầu vào
input_dir = r'D:\python_code\pythonProject\Coursera\XuLyAnh\image_data\NgoTruongGiang'

# Thư mục lưu ảnh khuôn mặt đã cắt
output_dir = r'D:\python_code\pythonProject\Coursera\XuLyAnh\crop'

# Duyệt qua tất cả các tập tin trong thư mục đầu vào
for filename in os.listdir(input_dir):
    # Đường dẫn đầy đủ đến tập tin ảnh
    img_path = os.path.join(input_dir, filename)

    # Đọc ảnh
    image = cv.imread(img_path)
    if image is None:
        continue  # Bỏ qua nếu không thể đọc ảnh

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Duyệt qua từng khuôn mặt phát hiện được
    for i, (x, y, w, h) in enumerate(faces_rect):
        # Cắt khuôn mặt
        face_roi = image[y:y + h, x:x + w]

        # Tạo tên tập tin cho khuôn mặt cắt
        face_filename = f'{os.path.splitext(filename)[0]}_face_{i}.jpg'
        face_output_path = os.path.join(output_dir, face_filename)

        # Lưu khuôn mặt đã cắt
        cv.imwrite(face_output_path, face_roi)


print('Done.')