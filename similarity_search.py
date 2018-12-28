import argparse, os, re

import tensorflow as tf
import numpy as np

from model.utils import Params
from model.alexnet import AlexNet
from PIL import Image
from validation_function2 import reli

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/checkpoint_data/',
                    help="Experiment directory to retrieve model checkpoints")
parser.add_argument('--model_config', default='experiments',
                    help="Experiment directory containing params.json")
parser.add_argument('--db_file', default='../../DeepHash/DeepHash/data/cifar10/fulltrain.txt',
                    help="Path to the file containing database data")
parser.add_argument('--query_file', default='../../DeepHash/DeepHash/data/cifar10/fulltest.txt',
                    help="Path to the file containing query data")
parser.add_argument('--data_dir', default='../../DeepHash/DeepHash/data/cifar10/',
                    help="Path to the folder containing data")


def __str_to_int(strings, classes):
    
    out = np.zeros(classes, dtype = np.int32)
    for i in range (classes):
        if strings[i] == '1':
            out[i] = 1

    return out


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_config, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)


    images = tf.placeholder( tf.float32 , [ None , params.image_size, params.image_size, params.num_channels] , name = 'images')
    labels = tf.placeholder( tf.int32 , [None , params.num_classes], name = 'labels' )
    alexnet_model = AlexNet(images, None, params, 'validate')

    # Values from layer fc7 and fclat
    emb_bin = tf.cast(tf.round(alexnet_model.fclat), tf.bool)
    emb_flt = alexnet_model.fc7

    database_embed_bin = []
    database_embed_float = []

    query_embed_bin = []
    query_embed_float = []

    with tf.Session() as sess:

        # Regex for parsing every line
        regex = re.compile('(?<=g )[0-1]( [0-1])*')

        # Restore saved model
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.model_dir))
        tf.logging.info("Forward propagating images through the DNN")
        
        with open(args.db_file, 'r') as db:
            # Array of db labels
            database_lab = []
            ims = []
            las = []
            ctr = 1
            for line in db:

                with Image.open(os.path.join(args.data_dir, line.split(' ')[0]), 'r') as im:
                    ima = np.resize(np.array(im), (params.image_size, params.image_size, params.num_channels))
                la = __str_to_int(re.search(regex, line)[0].split(" "), params.num_classes)
                
                ims.append(ima)
                las.append(la)

                if ctr%2 == 0:
                    flt, binn = sess.run([emb_flt, emb_bin], feed_dict = {images : ims, labels : las})

                    database_lab.extend(las)
                    database_embed_float.extend(flt.tolist())
                    database_embed_bin.extend(binn.tolist())

                    ims = []
                    las = []


        with open(args.query_file, 'r') as qu:
            # Array of query labels
            ims = []
            las = []
            ctr = 1
            query_lab = []
            for line in qu:

                with Image.open(os.path.join(args.data_dir, line.split(' ')[0]), 'r') as im:
                    ima = np.resize(np.array(im), (params.image_size, params.image_size, params.num_channels))
                la = __str_to_int(re.search(regex, line)[0].split(" "), params.num_classes)

                ims.append(ima)
                las.append(la)

                if ctr%2 == 0:
                    flt, binn = sess.run([emb_flt, emb_bin], feed_dict = {images : ims, labels : las})

                    query_lab.extend(las)
                    query_embed_float.extend(flt.tolist())
                    query_embed_bin.extend(binn.tolist())
                
                    ims = []
                    las = []

        database_embed_bin = np.array(database_embed_bin)
        database_embed_float = np.array(database_embed_float)
        database_lab = np.array(database_lab)

        query_embed_bin = np.array(query_embed_bin)
        query_embed_float = np.array(query_embed_float)
        query_lab = np.array(query_lab)
        
        print("Begining similarity search for", query_embed_bin.shape[0], "query images on" , database_embed_bin.shape[0], "database images")
        print("Similarity Search accuracy = ", reli(1, 50, query_embed_bin, query_embed_float, query_lab, database_embed_bin, database_embed_float, database_lab))