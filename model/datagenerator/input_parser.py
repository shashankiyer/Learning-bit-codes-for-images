"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

import re, os

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataParser(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    """

    def __init__(self, txt_file, data_dir, mode, params):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            data_dir: Path to the data folder
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            params: A dict of hyperparameters

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = params.num_classes
        self.image_size = params.image_size
        self.num_channels = params.num_channels

        # retrieve the data from the text file
        self.data_dir = data_dir
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        #if shuffle:
        #    self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls = params.num_parallel_calls)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls = params.num_parallel_calls)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        self.data = data

    def __str_to_int(strings):
        for i in range (10):
            if strings[i] == '1':
                out = i

        return out

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        mat = re.compile('(?<=g )[0-1]( [0-1])*')
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.img_paths.append(os.path.join(self.data_dir,line.split(' ')[0]))
                self.labels.append(ImageDataParser.__str_to_int(re.search(mat, line)[0].split(" ")))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels = self.num_channels)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # swap(2,1,0), bgr -> rgb
        im_rgb = tf.cast(img_centered, tf.float32)[:, :, ::-1]

        return im_rgb, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=self.num_channels)
        img_resized = tf.image.resize_images(img_decoded, [self.image_size, self.image_size])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # swap(2,1,0), bgr -> rgb
        im_rgb = tf.cast(img_centered, tf.float32)[:, :, :, ::-1]

        return im_rgb, one_hot
