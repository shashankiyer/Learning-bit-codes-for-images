import os

import numpy as np
import tensorflow as tf

from model.alexnet import AlexNet

"""
Configuration Part.
"""

def init_weights(alexnet_model):

    # loading from the .npy file
    def InitFn(scaffold,sess):
        alexnet_model.load_initial_weights(sess)
    return InitFn

def model_fn(features, labels, mode, params):
    """Model function for tf.estimator
    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    print(features.shape)
    # Unpack images
    images = features
    images = tf.reshape(images, [-1, params.image_size, params.image_size, 3])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    train_layers = ['fc8', 'fclat', 'fc7', 'fc6']
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        if is_training:
            alexnet_model = AlexNet(images, train_layers, params, 'train')
        else:
            alexnet_model = AlexNet(images, train_layers, params, 'validate')

    # Link variable to model output
    score = alexnet_model.fc8
    embeddings_bin = tf.round(alexnet_model.fclat)
    embeddings_floats = alexnet_model.fc7

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'Bit_codes': embeddings_bin}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Unpack labels
    labels = tf.cast(labels, tf.int32)

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Op for calculating the loss
    with tf.name_scope("softmax_cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                    labels=labels))

    # Train op
    with tf.name_scope("train"):

        gst = tf.train.create_global_step()

        # Get gradients of all trainable variables
        optimiser = tf.train.GradientDescentOptimizer(params.learning_rate)

        grads_and_vars = optimiser.compute_gradients(loss, var_list)

        fc6w_grad, _ = grads_and_vars[-8]
        fc6b_grad, _ = grads_and_vars[-7]
        fc7w_grad, _ = grads_and_vars[-6]
        fc7b_grad, _ = grads_and_vars[-5]
        fclatw_grad, _ = grads_and_vars[-4]
        fclatb_grad, _ = grads_and_vars[-3]
        fc8w_grad, _ = grads_and_vars[-2]
        fc8b_grad, _ = grads_and_vars[-1]

        # Create optimizer and apply gradient descent to the trainable variables
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('softmax_cross_entropy', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        prediction = tf.equal(tf.argmax(score, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(labels, predictions)
    }

    predictions_dict = {
        "predictions": predictions
    }

    export_outputs = {
        "emb_bin": tf.estimator.export.PredictOutput(predictions),
        "emb_float": tf.estimator.export.PredictOutput(embeddings_floats),
        "emb_lab": tf.estimator.export.PredictOutput(labels)
    }

    scaffold = tf.train.Scaffold(init_op=None,
                                    init_fn=init_weights(alexnet_model))

    return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions_dict,
                                        loss = loss, train_op = train_op,
                                        eval_metric_ops = eval_metric_ops, export_outputs = export_outputs, scaffold = scaffold)