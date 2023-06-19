import numpy as np
def apply_noise_single_image_bw(image, noise_amount, random: bool = False):
    x_shape = 784
    if random:
        noise_amount = np.random.random_sample()*0.5
    noise = np.random.normal(0.5, scale=noise_amount, size=(x_shape))
    noisy_image = image + noise
    
    return noisy_image


def apply_noise_to_data_bw(images, noise_amount, random: bool = False):
    noisy_images = []
    for image in images:
        noisy_images.append(apply_noise_single_image(
            image, noise_amount, random=random))
    noisy_images = np.asarray(noisy_images)
    return noisy_images


def add_artifact_to_img(img):
    artifact_size_x = np.random.randint(30, 40)
    artifact_size_y = np.random.randint(30, 40)
    artifact_x = np.random.randint(0, 128 - artifact_size_x)
    artifact_y = np.random.randint(0, 128 - artifact_size_y)

    artifact = img.copy()
    
    for i in range(artifact_size_x):
        for j in range(artifact_size_y):
            for d in range(3):
                artifact[artifact_x + i][artifact_y + j][d] = 255
    return np.clip(artifact, 0, 255)

def add_artifact_to_data(data):
    noisy_data = []
    for img in data:
        noisy_data.append(add_artifact_to_img(img))
    noisy_data = np.asarray(noisy_data)
    return noisy_data

def apply_noise_to_data_rgb(images):
    noisy_images = []
    for image in images:
        noisy_images.append(apply_noise_single_image_rgb(
            image))
    noisy_images = np.asarray(noisy_images)
    return noisy_images


def apply_noise_single_image_rgb(image):
    image_shape_x = image.shape[0]
    image_shape_y = image.shape[1]

    noise_r = np.random.normal(loc = 0, scale=122, size=(image_shape_x,image_shape_y))
    noise_g = np.random.normal(loc = 0, scale=122, size=(image_shape_x,image_shape_y))
    noise_b = np.random.normal(loc = 0, scale=122, size=(image_shape_x,image_shape_y))

    noisy_image = np.zeros((image_shape_x, image_shape_y,3))
    
    noisy_image[:,:,0] = image[:,:,0] + noise_r
    noisy_image[:,:,1] = image[:,:,1] + noise_g
    noisy_image[:,:,2] = image[:,:,2] + noise_b

    
    return np.clip(noisy_image,0,255)
