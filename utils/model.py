import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


IMAGE_SIZE = 256


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


class UNet(object):

    def __init__(self, input_channel_count, class_count):
        self.model_kernel = self.__make_model(input_channel_count, class_count)
        self.model_kernel.compile(loss=dice_coef_loss, optimizer=Adam(
            learning_rate=0.01), metrics=[dice_coef])

    def __make_model(self, input_channel_count, class_count):
        inputs = Input((IMAGE_SIZE, IMAGE_SIZE, input_channel_count))

        conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
        bn1 = Activation('relu')(conv1)
        conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
        bn1 = BatchNormalization(axis=3)(conv1)
        bn1 = Activation('relu')(bn1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

        conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
        bn2 = Activation('relu')(conv2)
        conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
        bn2 = BatchNormalization(axis=3)(conv2)
        bn2 = Activation('relu')(bn2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

        conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
        bn3 = Activation('relu')(conv3)
        conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
        bn3 = BatchNormalization(axis=3)(conv3)
        bn3 = Activation('relu')(bn3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

        conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
        bn4 = Activation('relu')(conv4)
        conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
        bn4 = BatchNormalization(axis=3)(conv4)
        bn4 = Activation('relu')(bn4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

        conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
        bn5 = Activation('relu')(conv5)
        conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
        bn5 = BatchNormalization(axis=3)(conv5)
        bn5 = Activation('relu')(bn5)

        up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(
            2, 2), padding='same')(bn5), conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), padding='same')(up6)
        bn6 = Activation('relu')(conv6)
        conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
        bn6 = BatchNormalization(axis=3)(conv6)
        bn6 = Activation('relu')(bn6)

        up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
            2, 2), padding='same')(bn6), conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), padding='same')(up7)
        bn7 = Activation('relu')(conv7)
        conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
        bn7 = BatchNormalization(axis=3)(conv7)
        bn7 = Activation('relu')(bn7)

        up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
            2, 2), padding='same')(bn7), conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), padding='same')(up8)
        bn8 = Activation('relu')(conv8)
        conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
        bn8 = BatchNormalization(axis=3)(conv8)
        bn8 = Activation('relu')(bn8)

        up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
            2, 2), padding='same')(bn8), conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), padding='same')(up9)
        bn9 = Activation('relu')(conv9)
        conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
        bn9 = BatchNormalization(axis=3)(conv9)
        bn9 = Activation('relu')(bn9)

        conv10 = Conv2D(class_count, (1, 1), activation='sigmoid')(bn9)

        return Model(inputs=[inputs], outputs=[conv10])


    def predict(self, x):
        return self.model_kernel.predict(x)


    def fit(self, x, y, batch_size=32, epochs=100, verbose=1):
        self.model_kernel.fit(x, y, batch_size=batch_size,
                              epochs=epochs, verbose=verbose)
        return self.model_kernel


    def load_weights(self, path):
        return self.model_kernel.load_weights(path)

