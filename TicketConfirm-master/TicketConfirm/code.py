from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import pandas as pd
from keras.models import load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

data = []
labels = []


input_size = 12

train = pd.read_csv("train.csv")
del train['FROM']
del train['TO']

label = train['CONFIRMED']
del train['CONFIRMED']
data = []

head = train.columns
for i,row in train.iterrows():
    temp = []
    for x in head:
        temp.append(train.at[i,x])
    data.append(temp)

data = np.array(data)

labels = []

for i in label:
    labels.append(i)

labels = np_utils.to_categorical(labels, len(set(labels)))

    #partition the data into training and testing splits, using 75%
    #of the data for training and the remaining 25% for the testing

print("[INFO] constructing trsining/testing split...")
    #(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size =0.25, random_state=42)

    #define the architecture of the neural network
model = Sequential()
model.add(Dense(int(12), input_dim=input_size, init="uniform", activation="relu"))
model.add(Dense(int(10), init="uniform", activation="relu"))
model.add(Dense(int(8), init="uniform", activation="relu"))
model.add(Dense(int(6),init="uniform", activation="relu"))
#model.add(Dense(int(4),init="uniform", activation="relu"))
#model.add(Dense(int(3),init="uniform", activation="relu"))
#model.add(Dense(int(2),init="uniform", activation="relu"))
#model.add(Dense(int(input_size*50), init="uniform", activation="relu"))
model.add(Dense(2, init="uniform", activation="relu"))
model.add(Activation("softmax"))

    #train model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(data, labels, nb_epoch=20)
#print("[INOF] evaluating result on testing set...")
#(loss, accuracy) = model.evaluate(data, labels, verbose=1)
#   print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
model.save('my_model')


test = pd.read_csv("test.csv")
data = []
del test['FROM']
del test['TO']

head = test.columns
for i,row in test.iterrows():
    temp = []
    for x in head:
        temp.append(test.at[i,x])
    data.append(temp)

data = np.array(data)
result = model.predict(data)


test['CONFIRMED']=0

for (i,row) in test.iterrows():
    test.at[i,'CONFIRMED'] = result[i][1]

test.to_csv("results.csv")
