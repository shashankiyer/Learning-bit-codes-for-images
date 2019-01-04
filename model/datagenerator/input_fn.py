from model.datagenerator.input_parser import ImageDataParser
import tensorflow as tf
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/cifar10/train.txt',
                    help="Path to the file containing training data")
parser.add_argument('--val_file', default='data/cifar10/test.txt',
                    help="Path to the file containing validation data")
parser.add_argument('--database_file', default='data/cifar10/test.txt',
                    help="Path to the file containing database data")                    
parser.add_argument('--query_file', default='data/cifar10/test.txt',
                    help="Path to the file containing database data")                    
parser.add_argument('--data_dir', default='data/cifar10/',
                    help="Path to the folder containing images")

args = parser.parse_args()              

def read_dataset(params, mode = tf.estimator.ModeKeys.TRAIN, db_flag = None):
  """ Input function that supplies data to the TF estimator
  """
  def _input_fn():
  
    if mode == tf.estimator.ModeKeys.TRAIN:
      which_file = args.train_file

    elif mode == tf.estimator.ModeKeys.EVAL:
      which_file = args.val_file

    elif db_flag is True:
      which_file = args.database_file
                                 
    else:
      which_file = args.query_file
                                                                 

    im_par = ImageDataParser(which_file,
                              args.data_dir,
                              mode,                    
                              params
                            )
    im_par.create_pipeline()                            

    tf.logging.info('inputs = {}'.format(im_par.data.output_shapes))

    return im_par.data

  return _input_fn

def get_labels_for_ss(params, db_flag):
  """ Input function that supplies labels for similarity search
  """
  if db_flag is True:
    which_file = args.database_file
                                 
  else:
    which_file = args.query_file

  mode = tf.estimator.ModeKeys.PREDICT
  im_par = ImageDataParser(which_file,
                              args.data_dir,
                              mode,
                              params
                            )

  num_images = len(im_par.labels)

  one_hot = np.zeros((num_images, params.num_classes), dtype = int)
  one_hot[np.arange(num_images), im_par.labels] = 1

  return one_hot, num_images
