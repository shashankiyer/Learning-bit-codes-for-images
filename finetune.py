import argparse
import os, shutil

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
'''
def experiment_fn(output_dir):
    # run experiment
    return tflearn.Experiment(
        tflearn.Estimator(model_fn=simple_rnn, model_dir=output_dir),
        train_input_fn=get_train(),
        eval_input_fn=get_valid(),
        eval_metrics={
            'rmse': tflearn.MetricSpec(
                metric_fn=metrics.streaming_root_mean_squared_error
            )
        }
    )
'''

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
    
    #shutil.rmtree(args.model_dir, ignore_errors=True) # start fresh each time

    train_spec = tf.estimator.TrainSpec(input_fn=get_train(params), max_steps=1000)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_valid(params))#, exporters=[exporter])

    estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir = args.model_dir, config = config, params = params)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Train the model
    #tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    #estimator.train(input_fn = get_train(params) , max_steps = 1)

    # Evaluate the model on the test set
    #tf.logging.info("Evaluation on test set.")
    #res = estimator.evaluate(input_fn = get_valid(params))
    #for key in res:
    #    print("{}: {}".format(key, res[key]))