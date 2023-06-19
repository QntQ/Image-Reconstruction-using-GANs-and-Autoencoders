import cv2
import numpy as np
import os
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from apply_noise import *

def normalize(data):
    return data/255


def denormalize(data):
    return data*255

def evaluate(model, testing, testing_noise, num_output):
    predicted = model.predict(testing_noise)
    print(predicted.shape)
    indices = np.random.randint(size=num_output, low=0, high=len(testing))
    error = 0
    for i in range(len(testing)):
        error = np.sum((predicted[i].astype("float") -
                        testing[i].astype("float")) ** 2)
        error /= len(testing)
    print(error)
    for i in indices:
        cv2.imwrite(f"./eval/testing{i}.jpg",
                    denormalize(testing[i].reshape(128, 128,3)))
        cv2.imwrite(f"./eval/predict{i}.jpg",
                    denormalize(predicted[i].reshape(128, 128,3)))
        cv2.imwrite(f"./eval/testing_noise{i}.jpg",
                    denormalize(testing_noise[i].reshape(128, 128,3)))



def input_preprocess(images):
    images_processed = add_artifact_to_img(images)
    return normalize(images_processed)


def generate_and_save_images(model, epoch, test_data, test_noise, history):
    test_data = test_data[:32]
    test_noise = test_noise[:32]
    indices = np.random.randint(size=3, low=0, high=len(test_data))
    error = 0
    test_noise = test_noise.reshape(len(test_noise), 128, 128, 3)
    predicted = model(test_noise, training=False).numpy()

    test_data = test_data.reshape(len(test_data), 128, 128, 3)

    for i in range(len(test_data)):
        error += np.sum((predicted[i] - test_data[i]) ** 2)
    error /= len(test_data)

    print(error)
    history.append(error)
    for i in indices:
        cv2.imwrite(f"./eval/testing{i}_epoch{epoch}.jpg",
                    denormalize(test_data[i]))
        cv2.imwrite(f"./eval/predict{i}_epoch{epoch}.jpg",
                    denormalize(predicted[i].reshape(128, 128, 3)))
        cv2.imwrite(f"./eval/testing_noise{i}_epoch{epoch}.jpg",
                    denormalize(test_noise[i]))
    return history
