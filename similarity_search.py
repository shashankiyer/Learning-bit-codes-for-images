import argparse, os

import tensorflow as tf
import numpy as np

from model.datagenerator.input_fn import read_dataset
from model.datagenerator.input_fn import get_labels_for_ss
from model.model_fn import model_fn
from model.utils import Params
from validation_function import reli

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/checkpoint_data',
                    help="Experiment directory to store model checkpoints")
parser.add_argument('--model_config', default='experiments',
                    help="Experiment directory containing params.json") 

def get_database(params):
  return read_dataset(params, mode=tf.estimator.ModeKeys.PREDICT, db_flag = True)

def get_query(params):
  return read_dataset(params, mode=tf.estimator.ModeKeys.PREDICT, db_flag = False)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_config, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the model
    tf.logging.info("Loading the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config) 

    # Compute embeddings for database images
    tf.logging.info("Acquiring Database Embeddings")
    predictions = estimator.predict(get_database(params))
    
    # Arrays to store db variables
    database_lab, samples = get_labels_for_ss(params, db_flag = True)
    database_embed_float = np.empty((samples, 4096))
    database_embed_bin = np.empty((samples, 48), dtype = bool)

    for i, p in enumerate(predictions):
        database_embed_float[i] = p['float_codes']
        database_embed_bin[i] = p['bit_codes']
    tf.logging.info("Done acquiring Database Embeddings")

    # Compute embeddings for database images
    tf.logging.info("Acquiring Query Embeddings")
    predictions = estimator.predict(get_query(params))

    # Arrays to store query variables
    query_lab, samples = get_labels_for_ss(params, db_flag = False)
    query_embed_float = np.empty((samples, 4096))
    query_embed_bin = np.empty((samples, 48), dtype = bool)

    for i, p in enumerate(predictions):
        query_embed_float[i] = p['float_codes']
        query_embed_bin[i] = p['bit_codes']
    tf.logging.info("Done acquiring Query Embeddings")

    print("Begining similarity search for", query_embed_bin.shape[0], "query images on" , database_embed_bin.shape[0], "database images")
    print("Similarity Search accuracy = %.2f" % reli(12, 120, query_embed_bin, query_embed_float, query_lab, database_embed_bin, database_embed_float, database_lab))