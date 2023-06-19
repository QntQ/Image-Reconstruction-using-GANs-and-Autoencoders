import os
import cv2
import numpy as np
import random
from apply_noise import *
from helpers import *
import asyncio
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images_mnist(mode: bool = True):
    # True for training, False for testing
    if mode:
        path = "MNIST - JPG - training"
    else:
        path = "MNIST - JPG - testing"

    images = []
    for i in range(1, 9):
        for j in os.listdir(path + "/" + str(i) + '/'):
            image = cv2.imread(path + "/" + str(i) + "/" + str(j))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten()
            images.append(image)
    return np.asarray(images)


def load_labels_mnist(mode: bool = True):
    # True for training, False for testing
    if mode:
        path = "MNIST - JPG - training"
    else:
        path = "MNIST - JPG - testing"

    labels = []
    for i in range(1, 9):
        for j in os.listdir(path + "/" + str(i) + '/'):
            labels.append(i)
    return labels


def get_data_mnist(mode: bool = True):
    # True for training, False for testing
    images = load_images(mode)
    labels = load_labels(mode)
    return images, labels


def get_ffhq_test_data():

    data_testing = []
    data_noise = []

    for j in os.listdir("./ffhq_dataset_testing/"):
        for i in os.listdir("./ffhq_dataset_testing/"+j):
            data_testing.append(cv2.imread("./ffhq_dataset_testing/"+j+"/"+i))
            data_noise.append(add_artifact_to_img(
                cv2.imread("./ffhq_dataset_testing/"+j+"/"+i)))

    return np.reshape(np.asarray(data_testing), (len(data_testing), 128*128*3)), np.reshape(np.asarray(data_noise), (len(data_noise), 128*128*3))


def keras_generator(batch_size):
    input_data_gen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=input_preprocess,
        horizontal_flip=True,)

    output_data_gen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=normalize,
        horizontal_flip=True,)

    seed = 1
    # input_data_gen.fit(images, augment=False, seed=seed)
    # output_data_gen.fit(images, augment=False, seed=seed)

    input_generator = input_data_gen.flow_from_directory(
        directory="./ffhq_dataset_training",
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )

    output_generator = output_data_gen.flow_from_directory(
        directory="./ffhq_dataset_training",
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )

    return input_generator, output_generator


def ffhq_gen(batch_size):

    folder = np.random.choice(os.listdir(
        "./ffhq_dataset_training"), 1)
    files = np.random.choice(os.listdir(
        f"./ffhq_dataset_training/{folder[0]}"), batch_size)
    images = []
    images_noise = []
    for i in files:
        images.append(normalize(cv2.imread(
            f"./ffhq_dataset_training/{folder[0]}/{i}")))
        images_noise.append(normalize(add_artifact_to_img(
            cv2.imread(f"./ffhq_dataset_training/{folder[0]}/{i}"))))
    return np.reshape(np.asarray(images), (batch_size, 128, 128, 3)), np.reshape(np.asarray(images_noise), (batch_size, 128, 128, 3))
