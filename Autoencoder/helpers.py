import cv2
import numpy as np
import os


def load_images(mode: bool = True):
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


def load_labels(mode: bool = True):
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


def get_data(mode: bool = True):
    # True for training, False for testing
    images = load_images(mode)
    labels = load_labels(mode)
    return images, labels


def apply_noise_single_image(image):
    x_shape = image.shape[0]
    noise_amount = np.random.uniform(0, 1)
    noise = np.random.normal(0, scale=noise_amount, size=(x_shape))
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def apply_noise_to_data(images):
    noisy_images = []
    for image in images:
        noisy_images.append(apply_noise_single_image(
            image))
    noisy_images = np.asarray(noisy_images)
    return noisy_images


def normalize(data):
    return data / 255


def denormalize(data):
    return data * 255


def add_artifact_to_img(img):
    artifact_size_x = np.random.randint(5, 10)
    artifact_size_y = np.random.randint(5, 10)
    artifact_x = np.random.randint(0, 28 - artifact_size_x)
    artifact_y = np.random.randint(0, 28 - artifact_size_y)

    artifact = img.copy()

    for i in range(artifact_size_x):
        for j in range(artifact_size_y):
            artifact[artifact_x + i][artifact_y + j] = 255
    return np.clip(artifact, 0, 255)


def add_artifact_to_data(data):
    noisy_data = []
    for img in data:
        noisy_data.append(add_artifact_to_img(img))
    noisy_data = np.asarray(noisy_data)
    return noisy_data


