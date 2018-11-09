

#importing keras lib
from keras.models import Sequential # use for creating the sequence of nodes and hidden layer
from keras.layers import Convolution2D # use convolution in pixel array of image
from keras.layers import MaxPooling2D # after that using MaxPooling for further reduction in size
from keras.layers import Flatten # then flatten the matrix for input in CNN 
from keras.layers import Dense 



# declare CNN classifier
classifier = Sequential()

# applying maxpooling and convolution twice to improve result
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) # using sigmoid for output in 0 and 1


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# traning the model 
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 2000)

classifier.save('model.nw') # saving the mode after it trains

# now we call the save model in another file  and use it for the prediction.
