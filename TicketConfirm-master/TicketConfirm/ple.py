import pandas as pd
import numpy as np
from tkinter import *
data = []
labels = []


input_size = 12

train = pd.read_csv("big_train.csv")
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

#labels = np_utils.to_categorical(labels, len(set(labels)))
X_train=data
y_train=labels
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
#
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13,13,13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

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
X_test=data
X_test = scaler.transform(X_test)

predictions = mlp.predict_proba(X_test)

root = Tk()
w = Label(root, text=predictions[0][1])
w.pack()
root.mainloop()