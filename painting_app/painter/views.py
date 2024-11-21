# painter/views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from datetime import datetime, timedelta
import math

# Load the header images
folderPath = os.path.join('painter', 'static', 'painter', 'Header')
myList = os.listdir(folderPath)
overlayList = [cv2.imread(os.path.join(folderPath, imPath)) for imPath in myList]

# Setting up Mediapipe and initial variables
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
width, height = 1280, 720
drawColor = (0, 0, 255)
thickness = 20
tipIds = [4, 8, 12, 16, 20]
xp, yp = [0, 0]
header = overlayList[0]
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Index page view
def index(request):
    return render(request, 'painter/index.html')

# def video():
#     #cap = cv2.VideoCapture('v4l2src device=/dev/video0 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
#     cap = cv2.VideoCapture(cv2.CAP_V4L2)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#     return cap

# Generator function to get video feed frames
def gen():
    global xp, yp, imgCanvas, header, drawColor, thickness, lastsave

    lastsave = datetime.now() - timedelta(seconds=5)

    cooldown_interval = timedelta(seconds=5)
    save_timestamp = None
    #cap = cv2.VideoCapture(cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
    #cap = cv2.VideoCapture(cv2.CAP_MSMF)
    #cap = cv2.VideoCapture('/dev/video0')

    if not cap.isOpened():
        print("Error: Could not open camera.")
    cap.set(3, width)
    cap.set(4, height)

    downloads_path = os.path.join(os.path.dirname(__file__), 'downloads')
    os.makedirs(downloads_path, exist_ok=True)

    with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Flip the image for a selfie-view and convert to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            # Convert back to BGR for OpenCV processing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if it's time to save the image without interrupting the video feed
            if save_timestamp and datetime.now() >= save_timestamp:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(downloads_path, f'drawing_{timestamp}.png')

                # Combine the current video frame with the drawing canvas
                imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
                imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
                saved_image = cv2.bitwise_and(image, imgInv)
                saved_image = cv2.bitwise_or(saved_image, imgCanvas)

                # Save the image to the downloads folder
                cv2.imwrite(filename, saved_image)
                print(f"Image saved as {filename}")
                save_timestamp = None  # Reset save timestamp after saving

            # Hand Landmark Detection
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]

                    if points:
                        x1, y1 = points[8]  # Index finger
                        x2, y2 = points[12] # Middle finger
                        x3, y3 = points[4]  # Thumb
                        x4, y4 = points[20]  # Pinky

                        # Determine which fingers are up
                        fingers = [
                            1 if points[tipIds[0]][0] < points[tipIds[0] - 1][0] else 0
                        ] + [
                            1 if points[tipIds[i]][1] < points[tipIds[i] - 2][1] else 0
                            for i in range(1, 5)
                        ]

                        # Selection Mode
                        if fingers[1] and fingers[2] and all(fingers[i] == 0 for i in [0, 3, 4]):
                            xp, yp = x1, y1
                            if y1 < 125:
                                if 0 < x1 < 143:
                                    header = overlayList[0]
                                    drawColor = (0, 0, 255)
                                elif 142 < x1 < 286:
                                    header = overlayList[1]
                                    drawColor = (255, 0, 0)
                                elif 285 < x1 < 432:
                                    header = overlayList[2]
                                    drawColor = (0, 255, 0)
                                elif 433 < x1 < 578:
                                    header = overlayList[3]
                                    drawColor = (128, 0, 128)
                                elif 579 < x1 < 726:
                                    header = overlayList[4]
                                    drawColor = (0, 255, 255)
                                elif 727 < x1 < 873:
                                    header = overlayList[5]
                                    drawColor = (0, 165, 255)
                                elif 874 < x1 < 1020:
                                    header = overlayList[6]
                                    drawColor = (255, 255, 0)
                                elif 1021 < x1 < 1164:
                                    header = overlayList[7]
                                    drawColor = (42, 42, 165)
                                elif 1164 < x1 < 1280:
                                    header = overlayList[8]
                                    drawColor = (0, 0, 0)
                            cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                        if all(fingers[i] == 0 for i in range(5)):
                            # The line between the index and the pinky indicates the Stand by Mode
                            cv2.line(image, (xp, yp), (x4, y4), drawColor, 5)
                            xp, yp = [x1, y1]

                        # Draw Mode
                        if fingers[1] and all(fingers[i] == 0 for i in [0, 2, 3, 4]):
                            cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED)
                            if xp == 0 and yp == 0:
                                xp, yp = x1, y1
                            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                            xp, yp = x1, y1

                        # Clear Canvas
                        if all(fingers[i] == 0 for i in range(5)):
                            imgCanvas = np.zeros((height, width, 3), np.uint8)
                            xp, yp = x1, y1

                        # Set Thickness
                        selecting = [1, 1, 0, 0, 0]
                        setting = [1, 1, 0, 0, 1]
                        if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(
                                fingers[i] == j for i, j in zip(range(0, 5), setting)):

                            # Getting the radius of the circle that will represent the thickness of the draw
                            # using the distance between the index finger and the thumb.
                            r = int(math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) / 3)

                            # Getting the middle point between these two fingers
                            x0, y0 = [(x1 + x3) / 2, (y1 + y3) / 2]

                            # Getting the vector that is orthogonal to the line formed between
                            # these two fingers
                            v1, v2 = [x1 - x3, y1 - y3]
                            v1, v2 = [-v2, v1]

                            # Normalizing it
                            mod_v = math.sqrt(v1 ** 2 + v2 ** 2)
                            v1, v2 = [v1 / mod_v, v2 / mod_v]

                            # Draw the circle that represents the draw thickness in (x0, y0) and orthogonaly
                            # translated c units
                            c = 3 + r
                            x0, y0 = [int(x0 - v1 * c), int(y0 - v2 * c)]
                            cv2.circle(image, (x0, y0), int(r / 2), drawColor, -1)

                            # Setting the thickness chosen when the pinky finger is up
                            if fingers[4]:
                                thickness = r
                                cv2.putText(image, 'Check', (x4 - 25, y4 - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0),
                                            1)
                            xp, yp = [x1, y1]

                        # To download
                        # Save Gesture
                        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                            if datetime.now() - lastsave >= cooldown_interval:
                                print("Save gesture detected")
                                
                                save_timestamp = datetime.now() + timedelta(seconds=5)
                                lastsave = datetime.now()
                                cv2.rectangle(image, (440, 310), (885, 366), (255, 255, 255), -1)
                                cv2.putText(image, f"Picture saved!",
                                            (width // 2 - 200, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                            2, (80, 175, 76), 3)

            # Overlay header and drawing
            image[0:125, 0:width] = header
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            image = cv2.bitwise_and(image, imgInv)
            image = cv2.bitwise_or(image, imgCanvas)

            # Encode frame for HTTP streaming
            ret, jpeg = cv2.imencode('.jpg', image)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

# Video feed view
def video_feed(request):
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')
