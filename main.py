import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import load as ld 
from utils import model


# trainフォルダ配下に画像imagesフォルダを置いている
FOLDER_PATH_X = './data/train' + os.sep + 'images'
# trainフォルダ配下にgt_imagesフォルダを置いている
FOLDER_PATH_Y = './data/train' + os.sep + 'gt_images'


def load_dataset(class_count):
    X_train, _ = ld.load_X(FOLDER_PATH_X)
    Y_train = ld.load_Y(FOLDER_PATH_Y, class_count)
    return X_train, Y_train


def main(input_channel_count,batch_size, epochs, class_count):

    X_train, Y_train = load_dataset(class_count)
    unet = model.UNet(input_channel_count, class_count)
    unet.fit(X_train, Y_train, batch_size = batch_size,
                        epochs = epochs, verbose=1)
    os.makedirs('output', exist_ok=True)
    unet.model_kernel.save_weights('output/unet_weights.hdf5')


if __name__ =="__main__":
    parser = argparse.ArgumentParser(
            prog='Image segmentation using U-Net',
            usage='python main.py',
            description='This module demonstrates image segmentation using U-Net.',
            add_help=True
        )
 
    parser.add_argument('-i', '--input_channel_count', type=int,
                        default=3, help='Input channel count')
    parser.add_argument('-e', '--epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=16, help='Batch size')
    parser.add_argument('-c', '--classcount', type=int,
                        default=4, help='Number of class')

    args = parser.parse_args()
    input_channel_count = args.input_channel_count
    epochs = args.epoch
    batch_size = args.batchsize
    class_count = args.classcount

    main(input_channel_count, batch_size, epochs, class_count)