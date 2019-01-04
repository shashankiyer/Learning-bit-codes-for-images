import os

import numpy as np
import tensorflow as tf

from model.alexnet import AlexNet


class RestoreHook(tf.train.SessionRunHook):
    """A TF hook to load initial weights from bvlc.npy
    """
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)

def model_fn(features, labels, mode, params):
    """Model function for tf.estimator
    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: dictionary of hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """

    # Unpack images
    images = features
    images = tf.reshape(images, [-1, params.image_size, params.image_size, 3])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    train_layers = ['fc8', 'fclat', 'fc7', 'fc6']
    
    alexnet_model = AlexNet(images, train_layers, params, mode)

    # Link variable to model output
    score = alexnet_model.fc8
    embeddings_bin = tf.cast(tf.round(alexnet_model.fclat), tf.bool)
    embeddings_float = alexnet_model.fc7
    
    # Creating a prediction dictionary
    predictions = {'bit_codes': embeddings_bin, 'float_codes': embeddings_float}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Cast labels
    labels = tf.cast(labels, tf.int32)

    # Op for calculating the loss
    with tf.name_scope("softmax_cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                    labels=labels))        

    # Model's predictions
    pred = tf.nn.softmax(score)

    eval_metric_ops = {
      "Evaluation_Accuracy": tf.metrics.accuracy(labels = tf.argmax(labels,1), predictions = tf.argmax(pred, 1))
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Train op
    with tf.name_scope("train"):

        gst = tf.train.get_or_create_global_step()

        optimiser = tf.train.GradientDescentOptimizer(params.learning_rate)

        # Get gradients of all trainable variables
        grads_and_vars = optimiser.compute_gradients(loss, var_list)

        fc6w_grad, _ = grads_and_vars[-8]
        fc6b_grad, _ = grads_and_vars[-7]
        fc7w_grad, _ = grads_and_vars[-6]
        fc7b_grad, _ = grads_and_vars[-5]
        fclatw_grad, _ = grads_and_vars[-4]
        fclatb_grad, _ = grads_and_vars[-3]
        fc8w_grad, _ = grads_and_vars[-2]
        fc8b_grad, _ = grads_and_vars[-1]

        # Apply gradient descent to the trainable variables
        train_op = optimiser.apply_gradients([(fc6w_grad, var_list[0]),
                                            (fc6b_grad, var_list[1]),
                                            (fc7w_grad, var_list[2]),
                                            (fc7b_grad, var_list[3]),
                                            (fclatw_grad, var_list[4]),
                                            (fclatb_grad, var_list[5]),
                                            (fc8w_grad, var_list[6]),
                                            (fc8b_grad, var_list[7])], global_step=gst)

    # Add gradients to summary
    for gradient, var in grads_and_vars:
        tf.summary.histogram(var.name[:-2] + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name[:-2], var)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correctness = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('Training_Accuracy', accuracy)

    # Assign pretrained weights to the AlexNet model
    init_fn = tf.contrib.framework.assign_from_values_fn(alexnet_model.get_map())

    return tf.estimator.EstimatorSpec(mode = mode, predictions = None,
                                        loss = loss, train_op = train_op,
                                        eval_metric_ops = None, training_hooks = [RestoreHook(init_fn)])