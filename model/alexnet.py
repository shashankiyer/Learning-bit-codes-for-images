import tensorflow as tf
import numpy as np
import os

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, skip_layer, params, mode):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            skip_layer: List of names of the layer, that get trained from
                scratch
            params: A dict of hyperparameters
        """

        # Parse input arguments into class variables
        self.X = x
        self.num_classes = params.num_classes
        self.KEEP_PROB = params.keep_prob
        self.SKIP_LAYER = skip_layer

        if mode == 'validate':
            self.KEEP_PROB = 1

        if os.path.exists('data/pretrained_alexnet/bvlc_alexnet.npy'):
            self.WEIGHTS_PATH = 'data/pretrained_alexnet/bvlc_alexnet.npy'
        else:
            raise ValueError("Couldn't locate model weights in data/pretrained_weights")

        # Dict that contains all the params of the DNN
        self.deep_params = {}

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, deep_params=self.deep_params, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        
        # 2nd Layer: Conv (w ReLu) -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, deep_params=self.deep_params, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, deep_params=self.deep_params, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, deep_params=self.deep_params, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, deep_params=self.deep_params, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        # Store these values to feedvalues.npy
        self.flattened = tf.reshape(pool5, [-1, 6*6*256])
        #Placeholder to feed the FC layers
        #self.flattened_placeholder = tf.placeholder(shape = tf.shape(self.flattened))
        fc6 = fc(self.flattened, 6*6*256, 4096, deep_params=self.deep_params, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        self.fc7 = fc(dropout6, 4096, 4096, deep_params=self.deep_params, name='fc7')
        dropout7 = dropout(self.fc7, self.KEEP_PROB)

        self.fclat = fc(dropout7, 4096, 48, deep_params=self.deep_params, name = 'fclat', relu=False, sigmoid=True)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(self.fclat, 48, self.num_classes, deep_params=self.deep_params, relu=False, name='fc8')


    def get_map(self):
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
        wd = {}
        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYER:
                for data in weights_dict[op_name]:
                    
                    # Biases
                    if len(data.shape) == 1:
                        wd[op_name + '/biases:0'] = data

                    # Weights
                    else:
                        wd[op_name + '/weights:0'] = data

        return wd

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, deep_params,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        deep_params[name] = [weights, biases]
        
    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, deep_params, relu=True, sigmoid=False):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', [num_out], initializer=tf.contrib.layers.xavier_initializer())

        deep_params[name] = [weights, biases]
        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu activation
        relu = tf.nn.relu(act)
        return relu
    elif sigmoid:
        # Apply Sigmoid activation
        sig = tf.nn.sigmoid(act)
        return sig
    else:
        # No activations
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
