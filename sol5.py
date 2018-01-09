from functools import lru_cache
import numpy as np
from scipy.misc import imread as imread
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from scipy import signal as signal
import matplotlib.pyplot as plt
from keras.layers import Input, Convolution2D, Activation, merge
from keras.models import Model
from keras.optimizers import Adam
import random
import sol5_utils

im_cache = {}
def read_image(filename, representation):
    """
    A method for reading an image from a path and loading in as gray or in color
    :param filename: The path for the picture to be loaded
    :param representation: The type of color the image will be load in. 1 for gray,
    2 for color
    :return: The loaded image
    """
    im = imread(filename)
    if representation == 1:
        # converting to gray
        im = rgb2gray(im) / 255
    else:
        if representation == 2:
            im = im.astype(np.float64)
            # setting the image's matrix to be between 0 and 1
            im = im / 255
    im_cache[filename] = im
    return im

def load_dataset(filenames, batch_size, corruption_func, crop_size):
    while True:
        # corrupted pics
        source_batch = np.zeros(shape=(batch_size, 1, crop_size[0], crop_size[1]), dtype=np.float64)
        # orig pics
        target_batch = np.zeros(shape=(batch_size, 1, crop_size[0], crop_size[1]), dtype=np.float64)
        random_pts = np.random.choice(len(filenames), batch_size)

        for i in range(len(random_pts)):
            isCached = im_cache.get(filenames[random_pts[i]], False)
            # checking cache
            if isCached is False:
                im = read_image(filenames[random_pts[i]], 1)
            else:
                im = im_cache[filenames[random_pts[i]]]
            # getting random crop top left pts based on dimension of pic
            crop_top_left_corner = [np.random.choice(im.shape[0] - crop_size[0], 1),
                                    np.random.choice(im.shape[1] - crop_size[1], 1)]
            org_corrupt = corruption_func(im)[int(crop_top_left_corner[0]):int(crop_top_left_corner[0]+crop_size[0]),
                         int(crop_top_left_corner[1]):int(crop_top_left_corner[1]+crop_size[1])]
            cropped_im = im[int(crop_top_left_corner[0]):int(crop_top_left_corner[0]+crop_size[0]),
                         int(crop_top_left_corner[1]):int(crop_top_left_corner[1]+crop_size[1])]
            source_batch[i][0] = org_corrupt
            target_batch[i][0] = cropped_im

            # plt.imshow(source_batch[i][0], cmap='gray')
            # plt.show()
            # plt.imshow(target_batch[i][0], cmap='gray')
            # plt.show()
        yield source_batch, target_batch

def resblock(input_tensor, num_channels):
    res = Convolution2D(num_channels, 3, 3, border_mode='same', dim_ordering='th')(input_tensor)
    res = Activation('relu')(res)
    res = Convolution2D(num_channels, 3, 3, border_mode='same', dim_ordering='th')(res)
    # TODO: check if merge/Add is the correct method for adding 2 tensors
    final = merge([input_tensor, res], mode='sum')
    return final

def build_nn_model(height, width, num_channels, num_res_blocks):
    a = Input(shape=(1, height, width))
    model = Convolution2D(num_channels, 3, 3, border_mode='same', dim_ordering='th')(a)
    model = Activation('relu')(model)
    residual_block = model
    for i in range(num_res_blocks):
        # getting res block after addition to input
        residual_block = resblock(residual_block, num_channels)
    # adding model before res blocks, to output of all chain
    # TODO: check if merge/Add is the correct method for adding 2 tensors
    model = merge([model, residual_block], mode='sum')
    model = Convolution2D(1, 3, 3, border_mode='same', dim_ordering='th')(model)
    return Model(input=a, output=model)



def train_model(model, images, corruption_func, batch_size,
                samples_per_epoch, num_epochs, num_valid_samples):
    # splitting the images to 80-20 batches
    valid_pts = np.random.choice(len(images), int(np.floor(0.8 * len(images))))
    test_pts = [i for i in range(len(images)) if i not in valid_pts]
    images = np.array(images)
    test_generator = load_dataset(images[test_pts], batch_size, corruption_func, (int(model.inputs[0].shape[2]), int(model.inputs[0].shape[3])))
    valid_generator = load_dataset(images[valid_pts], batch_size, corruption_func, (int(model.inputs[0].shape[2]), int(model.inputs[0].shape[3])))
    Adam(beta_2=0.9)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit_generator(test_generator, samples_per_epoch, validation_data=valid_generator, nb_epoch=num_epochs,
                        nb_val_samples=num_valid_samples)
    return model

def add_motion_blur(image, kernel_size, angle):
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return ndimage.filters.convolve(image, kernel, mode='nearest')

def helper_motion_blur(image):
    return random_motion_blur(image, [7])

def random_motion_blur(image, list_of_kernel_sizes):
    rand_angle = np.random.uniform(0, np.pi)
    #todo: change [7]
    return add_motion_blur(image, 7, rand_angle)

def restore_image(corrupted_image, base_model):
    a = Input(shape=(1, corrupted_image.shape[0], corrupted_image.shape[1]))
    b = base_model(a)
    new_model = Model(input=a, output=b)
    im = new_model.predict(np.reshape(np.array([corrupted_image]), (1, 1, corrupted_image.shape[0], corrupted_image.shape[1])))[0][0]
    return im

def strech_helper(im):
    """
    A helper function for stretching the image
    :param image: The input picture to be stretched
    :param hist: The histogram of the input image
    :return: The stretched image
    """
    return (im - np.min(im))/(np.max(im) - np.min(im))

#todo: remove default values in def signature of min max sigma
def add_gaussian_noise(image, min_sigma, max_sigma):
    div = random.uniform(min_sigma, max_sigma)
    # getting the normal noise, streching both the noise and the picture to values between 0-1
    normal_noise = strech_helper(np.random.normal(0, scale=div, size=(image.shape[0], image.shape[1])))
    image = strech_helper(image)
    noisy_img = (normal_noise + image) / 2
    return np.clip(noisy_img, 0, 1)


def helper_noise(im):
    return add_gaussian_noise(im, 0.5, 0.9)

import glob
image_list = []
for filename in glob.glob('C:\/Users\Imri\PycharmProjects\IP_ex5\ex5-imrilu\image_dataset\/train\*.jpg'):
    image_list.append(filename)

im = read_image(image_list[0], 1)
# plt.imshow(im, cmap='gray')
# plt.show()

# print(image_list)
#

images = sol5_utils.images_for_deblurring()

model = build_nn_model(16,16,32,5)
model = train_model(model, images, helper_motion_blur, 100, 1000, 3, 100)

specific = read_image(images[13], 1)

plt.imshow(specific, cmap='gray')
plt.show()
im2 = helper_motion_blur(specific)
plt.imshow(im2, cmap='gray')
plt.show()

cropped = restore_image(im2, model)
# cropped = strech_helper(cropped)

plt.imshow(cropped, cmap='gray')
plt.show()


# noisy = add_gaussian_noise(im, 0.88, 0.99)
# plt.imshow(noisy, cmap='gray')
# plt.show()
