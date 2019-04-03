# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import os
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

images=load_images_from_folder('./images/training/bird/')

# loop over the training images
for image in images:
	# load the image, convert it to grayscale, and describe it
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append('Passerine')
	data.append(hist)

images=load_images_from_folder('./images/training/non-bird/')

# loop over the training images
for image in images:
	# load the image, convert it to grayscale, and describe it
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append('Not a passerine')
	data.append(hist)

# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

images=load_images_from_folder('./images/testing')
# loop over the testing images
count=0
for image in images:
    # load the image, convert it to grayscale, describe it,
    # and classify it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    # display the image and the prediction
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,	1.0, (0, 0, 0), 2)
    cv2.imwrite("Test"+str(count)+".jpg", image)
    count=count+1