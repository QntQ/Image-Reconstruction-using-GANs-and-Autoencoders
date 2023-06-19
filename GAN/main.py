from train import *
from helpers import *
from apply_noise import *
from get_data import *
import matplotlib.pyplot as plt
from layer_utils import ReflectionPadding2D, res_block
from model_structure import *
import pickle
from config import *


generator = make_generator_model()
discriminator = make_discriminator_model()
combined = generator_containing_discriminator_multiple_outputs(
    generator, discriminator)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


#FFHQ is not included in the repo, but can be downloaded from https://github.com/QntQ/deep_autoencoder_mnist/blob/1973b95f4baef69ed891eb8d0442f40bdb242bb4/Autoencoder.py and copy into the "GAN" folder
test_data, test_noise = get_ffhq_test_data()

test_noise = normalize(test_noise)
test_data = normalize(test_data)


history, model = train((generator, discriminator,combined),EPOCHS,
                250, BATCH_SIZE, test_data, test_noise)

