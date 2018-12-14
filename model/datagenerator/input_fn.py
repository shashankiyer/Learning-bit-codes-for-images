from model.datagenerator.input_parser import ImageDataParser
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='../../DeepHash/DeepHash/data/cifar10/train.txt',
                    help="Path to the file containing training data")
parser.add_argument('--val_file', default='../../DeepHash/DeepHash/data/cifar10/test.txt',
                    help="Path to the file containing validation data")
parser.add_argument('--data_dir', default='../../DeepHash/DeepHash/data/cifar10/',
                    help="Path to the folder containing data")

args = parser.parse_args()              

def read_dataset(params, mode=tf.contrib.learn.ModeKeys.TRAIN):
  def _input_fn():
  
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      im_par = ImageDataParser(args.train_file,
                                  args.data_dir,
                                  'training',
                                  params
                                )
      _data = im_par.data                                
      # shuffle the first `buffer_size` elements of the dataset
      _data = _data.shuffle(buffer_size=params.buffer_size)
      # repeat elements as many times as epochs
      _data = _data.repeat(params.num_epochs)

    else:
      im_par = ImageDataParser(args.val_file,
                                  args.data_dir,
                                  'inference',
                                  params
                                )
      _data = im_par.data
    # create a new dataset with batches of images
    _data = _data.batch(params.batch_size)
    _data = _data.prefetch(1)

    print('inputs={}'.format(_data.output_shapes))

    return _data

  return _input_fn