import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imutils import paths
import os

data = []
labels = []

imagepaths = sorted(list(paths.list_images("..\Data\DigitDataset")))

for imgpath in imagepaths:
    image = cv2.imread(imgpath, 0)
    image = np.array(image)
    data.append(image)
    lab = imgpath.split(os.path.sep)[-2]
    labels.append(lab)
    
(traindata, testdata, trainlabel, testlabel) = train_test_split(data, labels, test_size = 0.2)

lb = preprocessing.LabelBinarizer()
trainlabel = lb.fit_transform(trainlabel)

traindata = np.array(traindata, np.float32)
trainlabel = np.array(trainlabel, np.float32)
testdata = np.array(testdata, np.float32)
testlabel = np.array(testlabel, np.float32)

traindata = traindata.reshape(-1, 784)
testdata = testdata.reshape(-1, 784)



ann = cv2.ml.ANN_MLP_create()

ann.setLayerSizes(np.array([784, 50, 10]))
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

ann.train(traindata, cv2.ml.ROW_SAMPLE, trainlabel)

ret, res = ann.predict(testdata)

result = np.argmax(res, axis=1)

matches = result == testlabel
correct = np.count_nonzero(matches)
acc = correct * 100/result.shape[0]
print("Total samples used for training: ", traindata.shape[0])
print("Total samples used for testing: ", testdata.shape[0])
print("Correctly classified samples: ", correct)
print("Accuracy: ", acc)
