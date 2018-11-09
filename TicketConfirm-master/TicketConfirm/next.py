import pandas as pd
import numpy as np
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

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 1)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(y_pred)
y_pred = (y_pred > 0.5)


test['CONFIRMED']=0

for (i,row) in test.iterrows():
    test.at[i,'CONFIRMED'] = y_pred[i]

test.to_csv("results.csv")