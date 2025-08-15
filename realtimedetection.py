import cv2
from keras.models import model_from_json
import numpy as np

# Load model architecture and weights
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocessing function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Use grayscale image for detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            img = extract_features(face_img)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(im, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

        cv2.imshow("Output", im)

        # Press Esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()
