import tensorflow as tf

def init_weights(alexnet_model):

    # loading from the .npy file

    def InitFn(scaffold,sess):
        alexnet_model.load_initial_weights(sess)
    return InitFn