import cv2 as cv

def find_working_camera_index():
    index = 0
    while True:
        cap = cv.VideoCapture(index)
        if not cap.isOpened():
            print(f"Cannot open camera at index {index}")
        else:
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {index} is working.")
                cap.release()
                break
            else:
                print(f"Camera at index {index} failed to return a frame.")
                cap.release()
        index += 1

find_working_camera_index()