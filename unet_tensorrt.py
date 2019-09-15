import os
import numpy as np # linear algebra
import pandas as pd # data processing
import cv2
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from skimage import io
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from tensorflow.python import pywrap_tensorflow
import os


K.set_learning_phase(0)
##data path


wkdir ='model/'
IMAGE_LIB = 'data/images/'
MASK_LIB = 'data/masks/'
IMG_HEIGHT, IMG_WIDTH = 256, 256
SEED = 42
smooth = 1e-12


all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.png']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH,3), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

all_images1 = [x for x in sorted(os.listdir(MASK_LIB)) if x[-4:] == '.png']

y_data = np.empty((len(all_images1), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images1):
    im = cv2.imread(MASK_LIB + name, 0).astype('float32')/255
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    y_data[i] = im

# fig, ax = plt.subplots(1,2, figsize = (8,4))
# ax[0].imshow(x_data[0], cmap='gray')
# ax[1].imshow(y_data[0], cmap='gray')
# plt.show()

y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.2)

print('Split train: ', len(x_train), len(y_train))

print('Split valid: ', len(x_val), len(y_val))

##jaccard coeffients function
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

##unet model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))
# 256

d0 = Conv2D(32, (3, 3), padding='same')(inputs)
d0 = BatchNormalization()(d0)
d0 = Activation('relu')(d0)
d0 = Conv2D(32, (3, 3), padding='same')(d0)
d0 = BatchNormalization()(d0)
d0 = Activation('relu')(d0)
d0_pool = MaxPooling2D((2, 2), strides=(2, 2))(d0)
# 128

d1 = Conv2D(64, (3, 3), padding='same')(d0_pool)
d1 = BatchNormalization()(d1)
d1 = Activation('relu')(d1)
d1 = Conv2D(64, (3, 3), padding='same')(d1)
d1 = BatchNormalization()(d1)
d1 = Activation('relu')(d1)
d1_pool = MaxPooling2D((2, 2), strides=(2, 2))(d1)
# 64

d2 = Conv2D(128, (3, 3), padding='same')(d1_pool)
d2 = BatchNormalization()(d2)
d2 = Activation('relu')(d2)
d2 = Conv2D(128, (3, 3), padding='same')(d2)
d2 = BatchNormalization()(d2)
d2 = Activation('relu')(d2)
d2_pool = MaxPooling2D((2, 2), strides=(2, 2))(d2)
# 32

d3 = Conv2D(256, (3, 3), padding='same')(d2_pool)
d3 = BatchNormalization()(d3)
d3 = Activation('relu')(d3)
d3 = Conv2D(256, (3, 3), padding='same')(d3)
d3 = BatchNormalization()(d3)
d3 = Activation('relu')(d3)
d3_pool = MaxPooling2D((2, 2), strides=(2, 2))(d3)
# 16

d4 = Conv2D(512, (3, 3), padding='same')(d3_pool)
d4 = BatchNormalization()(d4)
d4 = Activation('relu')(d4)
d4 = Conv2D(512, (3, 3), padding='same')(d4)
d4 = BatchNormalization()(d4)
d4 = Activation('relu')(d4)
d4_pool = MaxPooling2D((2, 2), strides=(2, 2))(d4)
# 8

d5 = Conv2D(1024, (3, 3), padding='same')(d4_pool)
d5 = BatchNormalization()(d5)
d5 = Activation('relu')(d5)
d5 = Conv2D(1024, (3, 3), padding='same')(d5)
d5 = BatchNormalization()(d5)
d5 = Activation('relu')(d5)
# center

up4 = UpSampling2D((2, 2))(d5)
up4 = concatenate([d4, up4], axis=3)
up4 = Conv2D(512, (3, 3), padding='same')(up4)
up4 = BatchNormalization()(up4)
up4 = Activation('relu')(up4)
up4 = Conv2D(512, (3, 3), padding='same')(up4)
up4 = BatchNormalization()(up4)
up4 = Activation('relu')(up4)
up4 = Conv2D(512, (3, 3), padding='same')(up4)
up4 = BatchNormalization()(up4)
up4 = Activation('relu')(up4)
# 16

up3 = UpSampling2D((2, 2))(up4)
up3 = concatenate([d3, up3], axis=3)
up3 = Conv2D(256, (3, 3), padding='same')(up3)
up3 = BatchNormalization()(up3)
up3 = Activation('relu')(up3)
up3 = Conv2D(256, (3, 3), padding='same')(up3)
up3 = BatchNormalization()(up3)
up3 = Activation('relu')(up3)
up3 = Conv2D(256, (3, 3), padding='same')(up3)
up3 = BatchNormalization()(up3)
up3 = Activation('relu')(up3)
# 32

up2 = UpSampling2D((2, 2))(up3)
up2 = concatenate([d2, up2], axis=3)
up2 = Conv2D(128, (3, 3), padding='same')(up2)
up2 = BatchNormalization()(up2)
up2 = Activation('relu')(up2)
up2 = Conv2D(128, (3, 3), padding='same')(up2)
up2 = BatchNormalization()(up2)
up2 = Activation('relu')(up2)
up2 = Conv2D(128, (3, 3), padding='same')(up2)
up2 = BatchNormalization()(up2)
up2 = Activation('relu')(up2)
# 64

up1 = UpSampling2D((2, 2))(up2)
up1 = concatenate([d1, up1], axis=3)
up1 = Conv2D(64, (3, 3), padding='same')(up1)
up1 = BatchNormalization()(up1)
up1 = Activation('relu')(up1)
up1 = Conv2D(64, (3, 3), padding='same')(up1)
up1 = BatchNormalization()(up1)
up1 = Activation('relu')(up1)
up1 = Conv2D(64, (3, 3), padding='same')(up1)
up1 = BatchNormalization()(up1)
up1 = Activation('relu')(up1)
# 128

up0 = UpSampling2D((2, 2))(up1)
up0 = concatenate([d0, up0], axis=3)
up0 = Conv2D(32, (3, 3), padding='same')(up0)
up0 = BatchNormalization()(up0)
up0 = Activation('relu')(up0)
up0 = Conv2D(32, (3, 3), padding='same')(up0)
up0 = BatchNormalization()(up0)
up0 = Activation('relu')(up0)
up0 = Conv2D(32, (3, 3), padding='same')(up0)
up0 = BatchNormalization()(up0)
up0 = Activation('relu')(up0)
up0 = Dropout(0.5)(up0)
# 256

classify = Conv2D(1, (1, 1), activation='sigmoid')(up0)

model = Model(inputs=inputs, outputs=classify)

model.compile(optimizer=Adam(1e-12), loss='binary_crossentropy', metrics=[jaccard_coef,'accuracy'])

model.summary()

def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


image_batch, mask_batch = next(my_generator(x_train, y_train, 8))
fix, ax = plt.subplots(8,2, figsize=(8,20))
# for i in range(8):
#     ax[i,0].imshow(image_batch[i,:,:,0])
#     ax[i,1].imshow(mask_batch[i,:,:,0])
# plt.show()

model.compile(optimizer=Adam(1e-12), loss='binary_crossentropy')

weight_saver = ModelCheckpoint('road_seg.h5', monitor='loss',
                                              save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)


results = model.fit_generator(my_generator(x_train, y_train, 8),
                           steps_per_epoch = 10,
                           validation_data = (x_val, y_val),
                           epochs=5, verbose=2,
                           callbacks = [weight_saver, annealer])

model.save(wkdir + 'unet_model.h5')

#2nd step
tf.keras.backend.set_learning_phase(0) #use this if we have batch norm layer in our network
from tensorflow.keras.models import load_model

# path we wanna save our converted TF-model
#MODEL_PATH = "./model/tensorflow/big/model1"
MODEL_PATH = "./save_model_ch"

# load the Keras model
#model = load_model('./model/modelLeNet5.h5')
model = load_model('./new_model_ch/road_seg.h5', custom_objects={'jaccard_coef':'accuracy'})

# save the model to Tensorflow model
saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
save_path = saver.save(sess, './data')

print("Keras model is successfully converted to TF model in "+MODEL_PATH)


# import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
#
# print_tensors_in_checkpoint_file(file_name='./save_model_ch/model-8500', tensor_name='', all_tensor_names=True, all_tensors= False)
''' 3rd step   '''


# import the needed libraries
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

# has to be use this setting to make a session for TensorRT optimization
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
    # import the meta graph of the tensorflow model
    # saver = tf.train.import_meta_graph("./model/tensorflow/big/model1.meta")
    saver = tf.train.import_meta_graph("./save_model_ch/data.meta")
    # then, restore the weights to the meta graph
    # saver.restore(sess, "./model/tensorflow/big/model1")
    saver.restore(sess, "./save_model_ch/data")

    # specify which tensor output you want to obtain
    # (correspond to prediction result)
    your_outputs = ["conv2d_28/Sigmoid"]

    # convert to frozen model
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,  # session
        tf.get_default_graph().as_graph_def(),  # graph+weight from the session
        output_node_names=your_outputs)
    # write the TensorRT model to be used later for inference
    with gfile.FastGFile("./new_model_ch/frozen_model.pb", 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")


# convert (optimize) frozen model to TensorRT model
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,  # frozen model
    outputs=your_outputs,
    max_batch_size=4,  # specify your max batch size
    max_workspace_size_bytes=(2 << 20),  # specify the max workspace
    precision_mode="INT8")  # precision, can be "FP32" (32 floating point precision) or "FP16"


#write the TensorRT model to be used later for inference
with gfile.FastGFile("./new_model_ch/TensorRT_model.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")




# check how many ops of the original frozen model
all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)

''' last '''
# import the needed libraries
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt # must import this although we will not use it explicitly
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt

# read the testing images (only for example)
img1= cv2.imread("data/images/vlcsnap-2019-02-14-11h40m32s789.png")
#print(img1_1.shape, img2_2.shape)
img1 = np.asarray(img1_1)

input_img = np.concatenate((img1.reshape((1, 256, 256, 3)),axis=0)

# function to read a ".pb" model
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
  with gfile.FastGFile(model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def




# variable
TENSORRT_MODEL_PATH = './new_model_ch/TensorRT_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1))) as sess:
        # read TensorRT model
        trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)

        # obtain the corresponding input-output tensor
        tf.import_graph_def(trt_graph, name='')

        input = sess.graph.get_tensor_by_name('input_1:0')
        output = sess.graph.get_tensor_by_name('conv2d_28/Sigmoid:0')

        # in this case, it demonstrates to perform inference for 50 times
        total_time = 0; n_time_inference = 50
        out_pred = sess.run(output, feed_dict={input: input_img})
        for i in range(n_time_inference):
            t1 = time.time()
            out_pred = sess.run(output, feed_dict={input: input_img})
            t2 = time.time()
            delta_time = t2 - t1
            total_time += delta_time
            print("needed time in inference-" + str(i) + ": ", delta_time)
        avg_time_tensorRT = total_time / n_time_inference
        print("average inference time: ", avg_time_tensorRT)






