from model.datagenerator.input_parser import ImageDataParser
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/cifar10/fulltrain.txt',
                    help="Path to the file containing training data")
parser.add_argument('--val_file', default='data/cifar10/fulltest1000.txt',
                    help="Path to the file containing validation data")
parser.add_argument('--data_dir', default='data/cifar10/',
                    help="Path to the folder containing data")

args = parser.parse_args()              

def read_dataset(params, mode=tf.contrib.learn.ModeKeys.TRAIN):
  """ Input function that supplies data to the TF estimator
    Args:
      params: dictionary of hyperparameters of the model (ex: `params.learning_rate`)
  """
  def _input_fn():
  
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      im_par = ImageDataParser(args.train_file,
                                  args.data_dir,
                                  'training',
                                  params
                                )

    else:
      im_par = ImageDataParser(args.val_file,
                                  args.data_dir,
                                  'evaluate',
                                  params
                                )

    tf.logging.info('inputs = {}'.format(im_par.data.output_shapes))

    return im_par.data

  return _input_fn


  