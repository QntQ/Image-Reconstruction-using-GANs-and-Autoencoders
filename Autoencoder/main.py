import keras.models
from Autoencoder import *
from helpers import *
import os
from tensorflow.keras.datasets import fashion_mnist

if __name__ == "__main__":
    
    # --- Load Data
    
    #FFHQ not included in this repo, if you want to try it out, download it from  https://github.com/NVlabs/ffhq-dataset
    (training, _) , (testing, _) = fashion_mnist.load_data()
    
    training_noise = apply_noise_to_data(training)
    testing_noise = apply_noise_to_data(testing)
    
    training = np.reshape(normalize(data=training), (60000, 28*28, 1))
    testing = np.reshape(normalize(data=testing), (10000, 28*28, 1))
    training_noise = np.reshape(normalize(data=training_noise), (60000, 28*28, 1))
    testing_noise = np.reshape(normalize(data=testing_noise), (10000, 28*28, 1))

    model = create_autoencoder_mnist()
    print("created model")
    
    print(model.summary())
    model = train_autoencoder(model, training_noise, training,
                              testing_noise, testing, epochs=500, batch_size=128)
    print("trained model")
    model.save("autoencoder.h5")

    print("evaluating model")
    eval_pics = []
    num_pics = 100

for i in range(num_pics):
    predict_img = model.predict(x=np.array([testing_noise[i]]))
    predict_img = denormalize(data=predict_img)
    test_img = denormalize(data=np.array([testing[i]]))
    cv2.imwrite("eval/" + str(i) + "_predict.jpg",
                predict_img.reshape(28, 28))
    cv2.imwrite("eval/" + str(i) + ".jpg",
                test_img.reshape(28, 28))
    cv2.imwrite("eval/" + str(i) + "_noise.jpg",
                testing_noise[i].reshape(28, 28))
