# -*- coding: utf-8 -*-
"""TF 2.0 Fast-SCNN.ipynb
"""

# !pip install tensorflow-gpu==2.0.0-alpha0

import tensorflow as tf
import os
import numpy as np
import cv2
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


"""
# Model Architecture
#### Custom function for conv2d: conv_block
"""

IMAGE_LIB = './data/images/'
MASK_LIB = './data/masks/'
IMG_HEIGHT, IMG_WIDTH = 512, 256
SEED=42
smooth = 1e-12

def my_model():


    def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
        if (conv_type == 'ds'):
            x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)
        else:
            x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)

        x = tf.keras.layers.BatchNormalization()(x)

        if (relu):
            x = tf.keras.activations.relu(x)

        return x


    """## Step 1: Learning to DownSample"""

    # Input Layer
    input_layer = tf.keras.layers.Input(shape=(512, 256, 3), name='input')

    lds_layer = conv_block(input_layer, 'conv', 4, (3, 3), strides=(2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 8, (3, 3), strides=(2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 16, (3, 3), strides=(2, 2))
    """## Step 2: Global Feature Extractor
    #### residual custom method
    """


    def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
        tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

        x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

        x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

        if r:
            x = tf.keras.layers.add([x, inputs])
        return x


    """#### Bottleneck custom method"""


    def bottleneck_block(inputs, filters, kernel, t, strides, n):
        x = _res_bottleneck(inputs, filters, kernel, t, strides)

        for i in range(1, n):
            x = _res_bottleneck(x, filters, kernel, t, 1, True)

        return x


    """#### PPM Method"""


    def pyramid_pooling_block(input_tensor, bin_sizes):
        concat_list = [input_tensor]
        w = 16
        h = 8

        for bin_size in bin_sizes:
            x = tf.keras.layers.AveragePooling2D(pool_size=(w // bin_size, h // bin_size),
                                                 strides=(w // bin_size, h // bin_size))(input_tensor)
            x = tf.keras.layers.Conv2D(32, 3, 2, padding='same')(x)
            x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h)))(x)

            concat_list.append(x)

        return tf.keras.layers.concatenate(concat_list)


    """#### Assembling all the methods"""

    gfe_layer = bottleneck_block(lds_layer, 16, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 24, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 32, (3, 3), t=6, strides=1, n=3)

    gfe_layer = pyramid_pooling_block(gfe_layer, [2, 4, 6, 8])

    """## Step 3: Feature Fusion"""
    ff_layer1 = conv_block(lds_layer, 'conv', 16, (1, 1), padding='same', strides=(1, 1), relu=False)

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(16, 1, 1, padding='same', activation=None)(ff_layer2)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)

    """## Step 4: Classifier"""

    classifier = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', strides=(1, 1), name='DSConv1_classifier')(
        ff_final)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', strides=(1, 1), name='DSConv2_classifier')(
        classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = conv_block(classifier, 'conv', 2, (1, 1), strides=(1, 1), padding='same', relu=False)

    classifier = tf.keras.layers.Dropout(0.3)(classifier)

    classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
    classifier = tf.keras.activations.softmax(classifier)
    """## Model Compilation"""

    fast_scnn = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)

    fast_scnn.summary()

    # tf.keras.utils.plot_model(fast_scnn, show_layer_names=True, show_shapes=True)

    fast_scnn.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return fast_scnn

# Training
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
#
# fig, ax = plt.subplots(1,2, figsize = (8,4))
# ax[0].imshow(x_data[0], cmap='gray')
# ax[1].imshow(y_data[0], cmap='gray')
# plt.show()

y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.2)

print('Split train: ', len(x_train), len(y_train))

print('Split valid: ', len(x_val), len(y_val))


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
for i in range(8):
    ax[i, 0].imshow(image_batch[i,:,:,0])
    ax[i, 1].imshow(mask_batch[i,:,:,0])
plt.show()

fast_scnn = my_model()
weight_saver = ModelCheckpoint('road_seg.h5', monitor='loss',
                                              save_best_only=True, save_weights_only=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

checkpoint_path = "training_1/fast_scnn_cp_per.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, )

results = fast_scnn.fit_generator(my_generator(x_train, y_train, 8),
                           steps_per_epoch = 200,
                           validation_data = (x_val, y_val),
                           epochs=25, verbose=2,callbacks = [cp_callback])


#Testing

model = my_model()

model.load_weights('training_1/fast_scnn_cp_per.ckpt')


def segment_road(img1):
    # input image
    #img1 = cv2.imread(file_name, 3)
    h, w, c = img1.shape
    img2 = cv2.resize(img1, dsize=(IMG_WIDTH, IMG_HEIGHT))
    img3 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

    # pridiction from UNET
    result = model.predict(img3.reshape(1, IMG_HEIGHT, IMG_WIDTH, -1))[0, :, :, 0]
    # saving the pridicted image
    io.imsave("testing_folder/unet_seg.jpg", result)
    io.imsave("testing_folder/img2_seg.jpg", img2)

    img3 = cv2.imread('testing_folder/unet_seg.jpg')
    img2gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)

    io.imsave("testing_folder/unet_seg1.jpg", mask)

    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(mask, kernel, iterations=3)
    erode1 = cv2.erode(mask, kernel, iterations=3)

    # # overlaping original image with mask image
    com_img = cv2.bitwise_and(img2, img2, mask=erode)
    com_img2 = cv2.bitwise_and(img2, img2, mask=erode)

    # cv2.imshow("img1", com_img)
    com_img[erode == 0] = [0, 0, 0]
    com_img2[erode1 == 0] = [255, 0, 0]
    com_img2 = cv2.resize(com_img2, (w, h))
    # com_img[erode != 0] = [0,255,255]
    # com_img = cv2.cvtColor(com_img, cv2.COLOR_BGR2RGB)

    frame = img1.copy()
    added_image = cv2.addWeighted(frame, 0.6, com_img2, 0.4, 0)
    added_image = cv2.resize(added_image, (w, h))

    # print("Processing even more images")
    cv2.imwrite("testing_folder/overlay.jpg", added_image)

    return added_image


if __name__ == '__main__':
    cap = cv2.VideoCapture('2019-02-20_12-18-45_out.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter('testing_folder/result.mp4', fourcc, 24, (1280, 720))
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]

    while (cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
           # print(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamps2 = (cap.get(cv2.CAP_PROP_POS_MSEC))
            img = segment_road(curr_frame)
            out1.write(img)
        else:
            break
    cap.release()
    out1.release()
    cv2.destroyAllWindows()


