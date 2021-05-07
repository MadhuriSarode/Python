# importing the required libraries
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
import cvlib as cvn
import random
import os
import glob

# loading the trained model
model = load_model('gender_detection.model')
classes = ['man', 'woman']

# load image files from the dataset -- test images
image_files = [f for f in glob.glob('/Users/madhuri/PycharmProjects/images/check' + "/*.jpg", recursive=True) if
               not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and displaying the gender detection for each image
for img in image_files:

    # reading the test image
    frame = cv.imread(img)

    # applying face detection
    face, confidence = cvn.detect_face(frame)

    # looping through detected faces
    for ind, f in enumerate(face):

        # getting corner points of face rectangle
        (X_start, Y_start) = f[0], f[1]
        (X_end, Y_end) = f[2], f[3]

        # drawing rectangle over face
        cv.rectangle(frame, (X_start, Y_start), (X_end, Y_end), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[Y_start:Y_end, X_start:X_end])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # applying gender detection on face
        conf = model.predict(face_crop)[0]

        # getting label of the predicted image with max accuracy
        ind = np.argmax(conf)
        label = classes[ind]

        label = "{}: {:.2f}%".format(label, conf[ind] * 100)

        Y = Y_start - 10 if Y_start - 10 > 10 else Y_start + 10

        # write label and confidence above face rectangle
        cv.putText(frame, label, (X_start, Y), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

    # displaying output with detected gender and its corresponding accuracy
    cv.imshow("gender detection", frame)
    cv.waitKey(0)
