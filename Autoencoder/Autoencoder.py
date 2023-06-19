from tensorflow.keras.utils import plot_model  # for plotting model diagram
from keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import Input  # for instantiating a keras tensor
from keras.models import Model  # for creating a Neural Network Autoencoder mode
from tensorflow import keras  # for building Neural Networks


print('Tensorflow/Keras: %s' % keras.__version__)  # print version


def create_autoencoder_ffhq():

    # --- Input Layer
    # Specify input shape
    x = Input(shape=(128, 128, 3), name='Input-Layer')

    e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
    batchnorm_1 = BatchNormalization()(pool1)

    e_conv2 = Conv2D(32, (3, 3), activation='relu',
                     padding='same')(batchnorm_1)
    pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
    batchnorm_2 = BatchNormalization()(pool2)

    e_conv3 = Conv2D(24, (3, 3), activation='relu',
                     padding='same')(batchnorm_2)
    pool3 = MaxPooling2D((2, 2), padding='same')(e_conv3)
    batchnorm_3 = BatchNormalization()(pool3)

    e_conv4 = Conv2D(16, (3, 3), activation='relu',
                     padding='same')(batchnorm_3)
    h = MaxPooling2D((2, 2), padding='same')(e_conv4)
    # Decoder  InvConv2D

    d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(d_conv1)

    d_conv2 = Conv2D(48, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(d_conv2)

    d_conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    up3 = UpSampling2D((2, 2))(d_conv3)

    d_conv4 = Conv2D(16, (3, 3), activation='relu', padding="same")(up3)
    up4 = UpSampling2D((2, 2))(d_conv4)

    r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up4)

    model = Model(x, r)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def create_autoencoder_mnist():
    
    n_inputs = 28*28
    
    # --- Input Layer
    # Specify input shape
    visible = Input(shape=(n_inputs,), name='Input-Layer')

    # --- Encoder Layer
    e = Dense(units=n_inputs/12, name='Encoder-Layer')(visible)
    e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
    e = LeakyReLU(name='Encoder-Layer-Activation')(e)

    # --- Middle Layer
    middle = Dense(units=n_inputs/24, activation='linear', activity_regularizer=keras.regularizers.L1(
        0.0001), name='Middle-Hidden-Layer')(e)

    # --- Decoder Layer
    d = Dense(units=n_inputs/12, name='Decoder-Layer')(middle)
    d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
    d = LeakyReLU(name='Decoder-Layer-Activation')(d)

    # --- Output layer
    output = Dense(units=n_inputs, activation='sigmoid',
                   name='Output-Layer')(d)

    # Define denoising autoencoder model
    model = Model(inputs=visible, outputs=output,
                  name='Denoising-Autoencoder-Model')

    # Compile denoising autoencoder model
    model.compile(optimizer='adam', loss='mse')
    return model


def train_autoencoder(autoencoder, x_train_noise, x_train, x_test_noise, x_test, epochs=1000, batch_size=128, verbose=1):
    autoencoder.fit(x_train_noise, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test_noise))
    return autoencoder
