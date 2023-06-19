from layer_utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Multiply, Lambda, Add, BatchNormalization, Activation, LeakyReLU, Conv2D, Conv2DTranspose

ngf = 64
input_nc = 3
output_nc = 3
n_blocks_gen = 6
# SGD

# crossentropy to mse


generator_optimizer = tf.keras.optimizers.SGD()
discriminator_optimizer = tf.keras.optimizers.SGD()


def make_generator_model():
    noise_shape = (128, 128, 3)
    """Build generator architecture."""
    # Current version : ResNet block
    inputs = Input(shape=noise_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Increase filter number
    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3),
                   strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # Apply 9 ResNet blocks
    mult = 2**n_downsampling
    for i in range(n_blocks_gen):
        x = res_block(x, ngf*mult, use_dropout=True)

    # Decrease filter number to 3 (RGB)
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(ngf * mult / 2),
                            kernel_size=(3, 3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='valid')(x)
    x = Activation('tanh')(x)

    # Add direct connection from input to output and recenter to [-1, 1]

    model = Model(inputs=inputs, outputs=x, name='Generator')
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.SGD())
    return model


ndf = 64
output_nc = 3
input_shape_discriminator = (128, 128, 3)


def make_discriminator_model():
    """Build discriminator architecture."""
    n_layers, use_sigmoid = 3, False

    inputs = Input(shape=input_shape_discriminator)

    x = Conv2D(filters=ndf, kernel_size=(4, 4),
               strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4),
                   strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4),
               strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])
    return model


def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    image_shape = (128, 128, 3)
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])

    model.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[
                  0.999, 0.001], optimizer=tf.keras.optimizers.SGD())
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
