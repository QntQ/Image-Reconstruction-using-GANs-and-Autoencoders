import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Multiply, Lambda, Add, BatchNormalization, Activation, LeakyReLU
from get_data import *
from config import *

def train_step(models, images, noise):
    # obtain a batch of data
    x_in, x_out = noise, images
    
    generator, discriminator, combined = models

    real_label, fake_label = np.ones(
        (BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))

    # Train on original images
    d_loss_real = discriminator.train_on_batch(x_in, fake_label)
    # Get batch of generated data from generator
    y_out = generator.predict(x_in)
    # Train on generated images
    for i in range(5):
        d_loss_gen = discriminator.train_on_batch(y_out, fake_label)
        d_loss_gen = discriminator.train_on_batch(x_out, real_label)

    main_hist = combined.fit(x_in, [x_out, real_label])
    return main_hist, (generator, discriminator, combined)


def train(models, epochs, steps_per_epoch, batch_size, test_data, test_noise):
    history = []
    for epoch in range(epochs):
        index = 0
        for _ in range(steps_per_epoch):
            i, j = ffhq_gen(batch_size)
            index += 1
            print("epoch:", epoch+1, "   step:", _)
            main_hist, models = train_step(models,i, j)
            history.append(main_hist)
    combined.save("GAN")
    return history, combined