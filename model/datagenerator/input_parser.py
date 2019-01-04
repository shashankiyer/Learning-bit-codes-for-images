"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

import re, os

class ImageDataParser(object):
    """Wrapper class around the Tensorflows dataset pipeline.
    """

    def __init__(self, txt_file, data_dir, mode, params):
        """Create a new ImageDataParser.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space and a 1-hot encoding, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train, val and pred

        Args:
            txt_file: Path to the text file.
            data_dir: Path to the data folder
            mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
            create_pipeline: flag to enable pipeline creation.
            params: Dict of hyperparameters
        Returns:
            tensorflow.data.Dataset: TF dataset for the estimator
        """
        self.IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

        self.txt_file = txt_file
        self.num_classes = params.num_classes
        self.image_size = params.image_size
        self.num_channels = params.num_channels

        self.num_parallel_calls = params.num_parallel_calls
        self.buffer_size = params.buffer_size
        self.num_epochs = params.num_epochs
        self.batch_size = params.batch_size

        self.mode = mode

        # retrieve the data from the text file
        self.data_dir = data_dir
        self._read_txt_file()

    def create_pipeline(self):
        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype = dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype = dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        data = data.map(self._parse_function, num_parallel_calls = self.num_parallel_calls)
            
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            # shuffle elements
            data = data.shuffle(buffer_size=self.buffer_size)
            # repeat elements as many times as epochs
            data = data.repeat(self.num_epochs)
            
        # create a new dataset with batches of images
        data = data.batch(self.batch_size)
        data = data.prefetch(1)

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

    def _parse_function(self, filename, label):
        """Input parser for all data samples."""
        
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        
        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels = self.num_channels)
        img_resized = tf.image.resize_images(img_decoded, [self.image_size, self.image_size])
        
        # Dataaugmentation.
        img_centered = tf.subtract(img_resized, self.IMAGENET_MEAN)

        # swap(2,1,0), bgr -> rgb
        im_rgb = tf.cast(img_centered, tf.float32)[:, :, ::-1]

        return im_rgb, one_hot
