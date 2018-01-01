from functools import lru_cache
import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as signal
import matplotlib.pyplot as plt
from keras.layers import Input, Convolution2D, Activation, add, merge
from keras import Model
from keras.optimizers import Adam
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
            cropped_im = im[int(crop_top_left_corner[0]):int(crop_top_left_corner[0]+crop_size[0]),
                         int(crop_top_left_corner[1]):int(crop_top_left_corner[1]+crop_size[1])]
            target_batch[i][0] = cropped_im
            source_batch[i][0] = corruption_func(cropped_im)

            # plt.imshow(source_batch[i][0], cmap='gray')
            # plt.show()
            # plt.imshow(target_batch[i][0], cmap='gray')
            # plt.show()
        yield source_batch, target_batch

def resblock(input_tensor, num_channels):
    res = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    res = Activation('relu')(res)
    res = Convolution2D(num_channels, 3, 3, border_mode='same')(res)
    # TODO: check if merge/Add is the correct method for adding 2 tensors
    final = merge([input_tensor, res], mode='sum')
    return final

def build_nn_model(height, width, num_channels, num_res_blocks):
    a = Input(shape=(1, height, width))
    model = Convolution2D(num_channels, 3, 3, border_mode='same')(a)
    model = Activation('relu')(model)
    residual_block = model
    for i in range(num_res_blocks):
        # getting res block after addition to input
        residual_block = resblock(residual_block, num_channels)
    # adding model before res blocks, to output of all chain
    # TODO: check if merge/Add is the correct method for adding 2 tensors
    model = merge([model, residual_block], mode='sum')
    model = Convolution2D(1, 3, 3, border_mode='same')(model)
    return Model(input=a, output=model)
def train_model(model, images, corruption_func, batch_size,
                samples_per_epoch, num_epochs, num_valid_samples):
    pass

def func(im):
    im[0,:] = 0
    return im

generator = load_dataset(["C:\ex1\gray_orig.png", "C:\ex1\pic2.jpg"], 1, func, (50,50))
generator.__next__()
generator.__next__()
a = im_cache.get("kaki", None)

# print(a)
# print(source.shape)
# print(target.shape)
build_nn_model(10,10,3,4)