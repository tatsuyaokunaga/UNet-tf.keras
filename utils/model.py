import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# 各種パラメターをグローバル変数として設定
default_unet_parameter = {
    "FIRST_LAYER_FILTER_COUNT": 64,
    "CONCATENATE_AXIS": -1,
    "CONV_FILTER_SIZE": 4,
    "CONV_STRIDE": 2,
    "CONV_PADDING": (1, 1),
    "DECONV_FILTER_SIZE": 2,
    "DECONV_STRIDE": 2
}

IMAGE_SIZE = 256

# ダイス係数(F値)を計算する関数
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)

# ロス関数
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


class UNet(object):

    def __init__(self, input_channel_count, class_count, parameter=default_unet_parameter):

        self.FIRST_LAYER_FILTER_COUNT = parameter["FIRST_LAYER_FILTER_COUNT"]
        self.CONCATENATE_AXIS = parameter["CONCATENATE_AXIS"]
        self.CONV_FILTER_SIZE = parameter["CONV_FILTER_SIZE"]
        self.CONV_STRIDE = parameter["CONV_STRIDE"]
        self.CONV_PADDING = parameter["CONV_PADDING"]
        self.DECONV_FILTER_SIZE = parameter["DECONV_FILTER_SIZE"]
        self.DECONV_STRIDE = parameter["DECONV_STRIDE"]

        self.model_kernel = self.__make_model(input_channel_count, class_count)
        self.model_kernel.compile(loss=dice_coef_loss, optimizer=Adam(
            learning_rate=0.01), metrics=[dice_coef])

    def __make_model(self, input_channel_count, class_count):
        # (256 x 256 x input_channel_count)
        inputs = Input((IMAGE_SIZE, IMAGE_SIZE, input_channel_count))

        # エンコーダーの作成
        # (128 x 128 x N)
        enc1 = Conv2D(self.FIRST_LAYER_FILTER_COUNT, self.CONV_FILTER_SIZE,
                      strides=self.CONV_STRIDE, padding="same")(inputs)

        # (64 x 64 x 2N)
        filter_count = self.FIRST_LAYER_FILTER_COUNT*2
        enc2 = self.__add_encoding_layer(filter_count, enc1)

        # (32 x 32 x 4N)
        filter_count = self.FIRST_LAYER_FILTER_COUNT*4
        enc3 = self.__add_encoding_layer(filter_count, enc2)

        # (16 x 16 x 8N)
        filter_count = self.FIRST_LAYER_FILTER_COUNT*8
        enc4 = self.__add_encoding_layer(filter_count, enc3)

        # (8 x 8 x 8N)
        enc5 = self.__add_encoding_layer(filter_count, enc4)

        # (4 x 4 x 8N)
        enc6 = self.__add_encoding_layer(filter_count, enc5)

        # (2 x 2 x 8N)
        enc7 = self.__add_encoding_layer(filter_count, enc6)

        # (1 x 1 x 8N)
        enc8 = self.__add_encoding_layer(filter_count, enc7)

        # デコーダーの作成
        # (2 x 2 x 8N)
        dec1 = self.__add_decoding_layer(filter_count, enc8)
        dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
        dec2 = self.__add_decoding_layer(filter_count, dec1)
        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (8 x 8 x 8N)
        dec3 = self.__add_decoding_layer(filter_count,  dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

        # (16 x 16 x 8N)
        dec4 = self.__add_decoding_layer(filter_count, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

        # (32 x 32 x 4N)
        filter_count = self.FIRST_LAYER_FILTER_COUNT*4
        dec5 = self.__add_decoding_layer(filter_count, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 2N)
        filter_count = self.FIRST_LAYER_FILTER_COUNT*2
        dec6 = self.__add_decoding_layer(filter_count, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x N)
        filter_count = self.FIRST_LAYER_FILTER_COUNT
        dec7 = self.__add_decoding_layer(filter_count, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x output_channel_count)
        dec8 = Activation(activation='swish')(dec7)
        out = Conv2DTranspose(
            class_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        out = Activation(activation='sigmoid')(out)

        return Model(inputs=inputs, outputs=out)

    def __add_encoding_layer(self, filter_count, sequence):
        new_sequence = Activation(activation='swish')(sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE,
                              strides=self.CONV_STRIDE, padding="same")(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def __add_decoding_layer(self, filter_count, sequence):
        new_sequence = Activation(activation='swish')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform', padding="same")(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)

        return new_sequence

    def predict(self, x):
        return self.model_kernel.predict(x)

    def fit(self, x, y, batch_size=32, epochs=100, verbose=1):
        self.model_kernel.fit(x, y, batch_size=batch_size,
                              epochs=epochs, verbose=verbose)
        return self.model_kernel

    def load_weights(self, path):
        return self.model_kernel.load_weights(path)

