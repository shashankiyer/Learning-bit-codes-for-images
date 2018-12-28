import argparse
import os

import tensorflow as tf

from model.datagenerator.input_fn import read_dataset 
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/checkpoint_data',
                    help="Experiment directory to store model checkpoints")
parser.add_argument('--model_config', default='experiments',
                    help="Experiment directory containing params.json")

def get_train(params):
  return read_dataset(params, mode=tf.contrib.learn.ModeKeys.TRAIN)

def get_valid(params):
  return read_dataset(params, mode=tf.contrib.learn.ModeKeys.EVAL)

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_config, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)

    # Train spec
    train_spec = tf.estimator.TrainSpec(input_fn=get_train(params), max_steps=1000)

    # Eval spec
    eval_spec = tf.estimator.EvalSpec(input_fn=get_valid(params))

    estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir = args.model_dir, config = config, params = params)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)