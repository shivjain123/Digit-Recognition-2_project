import cv2
import numpy as np
import pandas as pd
import PIL.ImageOps
import os
import ssl
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acs
from PIL import Image

if (not os.environ.get('PYTHONHTTPSVERIFIED', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
Y = pd.read_csv("labels.csv")["labels"]
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

x_train, x_test, y_train, y_test = tts(X, Y, train_size=7500, test_size=2500, random_state=42)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

lr = LogisticRegression(solver='saga', multi_class='multinomial')

lr.fit(x_train_scaled, y_train)

y_pred = lr.predict(x_test_scaled)
ac_score = acs(y_pred, y_test)

print(f"The Accuracy Score by Logistic Regression is {ac_score}.")

video = cv2.VideoCapture(0)

while (True):
    ret, frame = video.read()
    print(ret)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    upper_left = (int(width/2 - 56), int(height/2 - 56))
    bottom_right = (int(width/2 + 56), int(height/2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
    aoi = gray[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
    iam_pil = Image.fromarray(aoi)
    image_bw = iam_pil.convert('L').resize((22, 30), Image.ANTIALIAS)
    image_inverted = PIL.ImageOps.invert(image_bw)
    pixel_filter = 20
    min_pixel = np.percentile(image_inverted, pixel_filter)
    image_scaled = np.clip(image_inverted - min_pixel, 0, 255)
    max_pixel = np.max(image_inverted)
    image_scaled = np.asarray(image_scaled)/max_pixel
    test_sample = np.array(image_scaled).reshape(1, 660)
    test_pred = lr.predict(test_sample)
    print("Predicted class is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
