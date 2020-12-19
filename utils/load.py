import os
import re

import numpy as np
from PIL import Image


IMAGE_SIZE = 256
PATTERN = ".*\.(jpg|png|bmp)"


def normalize_x(image):
    image = image/127.5 - 1
    return image

# インプット画像を読み込む関数
def load_X(folder_path):
    import os
    import cv2

    image_files = [f for f in os.listdir(folder_path) if re.search(PATTERN, f)]
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE,
                       IMAGE_SIZE, 3), np.float32)
    for i, image_file in enumerate(image_files):
        image = Image.open(folder_path + os.sep + image_file)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        image = np.array(image)
        images[i] = normalize_x(image)
    return images, image_files


# ラベル画像を読み込む関数
def load_Y(folder_path, class_number):
    image_files = [f for f in os.listdir(folder_path) if re.search(PATTERN, f)]
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE,
                       IMAGE_SIZE, class_number), np.float32)
    for i, image_file in enumerate(image_files):
        image = Image.open(folder_path + os.sep + image_file)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image, dtype=np.uint8)
        # インデックス値をone-hotベクトルに直す
        identity = np.identity(class_number, dtype=np.uint8)  # 単位行列を生成
        images_segmented = identity[image]
        images[i] = images_segmented
    return images


def load_data(folder_path):
    image_files = [f for f in os.listdir(folder_path) if re.search(PATTERN, f)]
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE), np.float32)
    for i, image_file in enumerate(image_files):
        image = Image.open(folder_path + os.sep + image_file)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image, dtype=np.uint8)
        images[i] = image
    return images
