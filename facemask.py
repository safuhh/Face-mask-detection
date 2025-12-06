# -*- coding: utf-8 -*-
"""
Live Face Mask Detection - Working on Python 3.13 + TF 2.11+
"""

import cv2
import numpy as np
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# ----------------------------
# BUILD MODEL (if training from scratch)
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# DATA GENERATORS
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(150,150),
    batch_size=16,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(150,150),
    batch_size=16,
    class_mode='binary'
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
# Use fit() instead of deprecated fit_generator()
model.fit(
    training_set,
    epochs=10,
    validation_data=test_set
)

# Save trained model
model.save('mymodel.h5')

# ----------------------------
# LOAD MODEL FOR LIVE DETECTION
# ----------------------------
mymodel = load_model('mymodel.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (150,150))
        face_array = img_to_array(face_img_resized)
        face_array = np.expand_dims(face_array/255.0, axis=0)

        pred = mymodel.predict(face_array)[0][0]

        if pred >= 0.5:
            label = 'NO MASK'
            color = (0,0,255)
        else:
            label = 'MASK'
            color = (0,255,0)

        cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, str(datetime.datetime.now()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Live Mask Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
