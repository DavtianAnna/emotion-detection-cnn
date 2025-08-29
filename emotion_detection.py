import os
import cv2
import pickle
import numpy as np 

labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera ＞﹏＜")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame ＞﹏＜")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (48, 48))
    img_input = np.reshape(img_resized, (1, 48, 48, 1))
    img_input = img_input / 255.0 

    y_pred = model.predict(img_input)
    emotion = labels[np.argmax(y_pred)]
    prob = np.max(y_pred)

    text = f'{emotion} ({prob:.2f})'
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
