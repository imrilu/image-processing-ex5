import numpy as np
from scipy.misc import imread as imread
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from keras.layers import Input, Convolution2D, Activation, merge
from keras.models import Model
from keras.optimizers import Adam
import random
import sol5_utils


im_cache = {}
corrupt_cache = {}

def read_image(filename, representation=1):
    """
    A method for reading an image from a path and loading in as gray or in color
    :param filename: The path for the picture to be loaded
    :param representation: The type of color the image will be load in. 1 for gray,
    2 for color
    :return: The loaded image
    """
    im = imread(filename)
    if representation == 1:
        im = rgb2gray(im)
    elif representation == 2:
        im = (im / 255).astype(np.float32)
    im = im.astype(np.float64)
    return im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator method for yielding batches of corrupted/original image patches of input size
    :param filenames: The paths of image files to use
    :param batch_size: The size of each batch in the generator
    :param corruption_func: the corruption function to use
    :param crop_size: the size of each patch we extract from the image
    :return: generator, yielding each time a set of corrupted/original image patches
    """
    while True:
        height, width = crop_size
        # creating the array for the source and target batches
        source_batch = np.empty((batch_size, 1, height, width))
        target_batch = np.empty((batch_size, 1, height, width))
        for i in range(batch_size):
            file = random.choice(filenames)
            if file not in im_cache.keys():
                im_cache[file] = read_image(file)
                corrupt_cache[file] = corruption_func(im_cache[file])
            # choosing the top left corner of the patch to extract
            rand_col = random.randint(0, im_cache[file].shape[1] - width)
            rand_row = random.randint(0, im_cache[file].shape[0] - height)
            target_batch[i, 0, :, :] = im_cache[file][rand_row:rand_row + height,
                                       rand_col: rand_col + width] - 0.5
            source_batch[i, 0, :, :] = corrupt_cache[file][rand_row:rand_row + height,
                                       rand_col: rand_col + width] - 0.5

        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    """
    A method for creating 1 ResNet block
    :param input_tensor: the input tensor we use in our model
    :param num_channels: number of channels for the model
    :return: a ResNet block
    """
    res = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    res = Activation('relu')(res)
    res = Convolution2D(num_channels, 3, 3, border_mode='same')(res)
    # TODO: check if merge/Add is the correct method for adding 2 tensors
    final = merge([input_tensor, res], mode='sum')
    return final


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    A method for creating a model
    :param height: The height of the input to the model
    :param width: The width of the input to the model
    :param num_channels: number of channels for the model
    :param num_res_blocks: number of ResNet blocks that will be in the model
    :return: A model as seen in pdf
    """
    a = Input(shape=(1, height, width))
    model = Convolution2D(num_channels, 3, 3, border_mode='same')(a)
    model = Activation('relu')(model)
    residual_block = model
    for i in range(num_res_blocks):
        # getting res block after addition to input
        residual_block = resblock(residual_block, num_channels)
    # adding model before res blocks, to output of all chain
    model = merge([model, residual_block], mode='sum')
    model = Convolution2D(1, 3, 3, border_mode='same')(model)
    return Model(input=a, output=model)


def train_model(model, images, corruption_func, batch_size,
                samples_per_epoch, num_epochs, num_valid_samples):
    """
    A method for training a model on a set of images
    :param model: The model to train
    :param images: The batch of pictures to train on
    :param corruption_func: The corruption function we use to train the network
    :param batch_size: The size of each batch in the generator
    :param samples_per_epoch: The number of samples per epochs
    :param num_epochs: The number of epochs to use when training the model
    :param num_valid_samples: The number of samples to use in validation phase
    """
    # splitting the images to 80-20 batches
    valid_pts = np.random.choice(len(images), int(np.floor(0.8 * len(images))))
    test_pts = [i for i in range(len(images)) if i not in valid_pts]
    images = np.array(images)
    test_generator = load_dataset(images[test_pts], batch_size, corruption_func, (int(model.inputs[0].shape[2]), int(model.inputs[0].shape[3])))
    valid_generator = load_dataset(images[valid_pts], batch_size, corruption_func, (int(model.inputs[0].shape[2]), int(model.inputs[0].shape[3])))
    model.compile(optimizer=Adam(beta_2=0.9), loss='mean_squared_error')
    model.fit_generator(test_generator, samples_per_epoch, validation_data=valid_generator, nb_epoch=num_epochs,
                        nb_val_samples=num_valid_samples)


def add_motion_blur(image, kernel_size, angle):
    """
    A method for adding a motion blur effect to the image
    :param image: the image to blur
    :param kernel_size: the size of the kernel with which we convolve
    :param angle: the angle of the motion blur to perform
    :return: a blurred picture
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return ndimage.filters.convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    A method for using motion blur on a pic with random angle
    :param image: The image to blur
    :param list_of_kernel_sizes: A list of odd integers. we'll sample a kernel from the list randomly and
    use it when performing the convolution.
    :return: A blurred picture
    """
    rand_angle = np.random.uniform(0, np.pi)
    kernel_size = int(np.random.choice(list_of_kernel_sizes, 1))
    return add_motion_blur(image, kernel_size, rand_angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    A method for learning a model for deblurring images
    :param num_res_blocks: the number of ResNet blocks to use when constructing the model
    :param quick_mode: A boolean indicator for training in quick mode
    :return: A fully trained model
    """
    images = sol5_utils.images_for_deblurring()
    model = build_nn_model(16, 16, 32, num_res_blocks)
    lambda_blur = lambda x: random_motion_blur(x, [7])
    if quick_mode:
        train_model(model, images, lambda_blur, 10, 30, 2, 30)
    else:
        train_model(model, images, lambda_blur, 100, 10000, 10, 1000)
    return model


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    A function for adding a gaussian noise to a given picture
    :param image: The image to add gaussian noise to
    :param min_sigma: The min sigma range from which we will randomly sample a value
    :param max_sigma: The max sigma range from which we will randomly sample a value
    :return: A noisy picture
    """
    div = random.uniform(min_sigma, max_sigma)
    # getting the normal noise, streching both the noise and the picture to values between 0-1
    normal_noise = np.random.normal(0, scale=div, size=(image.shape[0], image.shape[1])).astype(np.float32)
    noisy_img = normal_noise + image
    return np.clip(noisy_img, 0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    A method for learning a model for denoising pictures
    :param num_res_blocks: The number of ResNet blocks to be in the trained model
    :param quick_mode: A boolean to indicate if we're on quick mode
    :return: A trained model for denoising pictures
    """
    images = sol5_utils.images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    lambda_noise = lambda x: add_gaussian_noise(x, 0, 0.2)
    if quick_mode:
        train_model(model, images, lambda_noise, 10, 30, 2, 30)
    else:
        train_model(model, images, lambda_noise, 100, 10000, 5, 1000)
    return model


def restore_image(corrupted_image, base_model):
    """
    A method for restoring a corrupted image with a given base model
    :param corrupted_image: The corrupted image to restore
    :param base_model: The base model to use to restore the image
    :return: The restored image
    """
    a = Input(shape=(1, corrupted_image.shape[0], corrupted_image.shape[1]))
    b = base_model(a)
    new_model = Model(input=a, output=b)
    corrupted_image = corrupted_image[np.newaxis, np.newaxis, :, :] - 0.5
    restored_img = new_model.predict(corrupted_image)[0].astype(np.float64)
    restored_img += 0.5
    return restored_img.clip(0, 1).reshape(corrupted_image.shape[2], corrupted_image.shape[3])

