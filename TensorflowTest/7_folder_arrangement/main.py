# The shell APIs
# can be further split into train.py and infer.py, but two APIs are more preferred
import os
import json
import random
import sys
import tensorflow as tf
import time

# from models.make_linear_regression_model import make_model      # helper function to load any models you have
from models.make_logistic_regression_model import make_model
# from models.make_nearest_neighbour_model import make_model
# from hpsearch import hyperband, randomsearch

# make my paths absolute to be independent from where python binary is called
dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

# general
flags.DEFINE_string('model_name', 'logistic_regression', 'name the model')
flags.DEFINE_string('best', None, 'best model achieved so far')

# HP search config
flags.DEFINE_boolean('fullsearch', False,
                     'Perform a full search of HP space ex:(hyperband -> lr search -> hyperband with best lr)')
flags.DEFINE_boolean('dry_run', False, 'Perform a dry_run (testing purpose)')
flags.DEFINE_integer('nb_process', 1, 'Number of parallel process to perform HP search')    # e.g. 4 default

# when doing hyperband or randomsearch
# fix some of the HPs inside model, e.g. explore diff models fixing the lr
flags.DEFINE_string('fixed_params', '{}',       # apply when doing HP search
                    'JSON inputs to fix some params in a HP search, ex \'{"lr": 0.001}\'')

# agent config
flags.DEFINE_boolean('debug', True, 'Debug mode')
flags.DEFINE_integer('max_iter', 100, 'Number of training steps')
flags.DEFINE_float('lr', 0.01, 'learning rate')
flags.DEFINE_integer('bsize', 32, 'batch size')
flags.DEFINE_integer('nb_units', 1, 'Number of hidden nodes')
flags.DEFINE_boolean('infer', False, 'Load an agent for playing')       # False: training first, True: testing/inferring

# important for TensorBoard
# choose to name the output folder
flags.DEFINE_integer('save_every', 5, 'save model and intermediate results for every <> epochs')
# fresh new directory by time.time()
#   1) for training
flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + str(int(time.time())),
                    'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')
# #   2) for testing/inferring
# flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + '1495532417',
#                     'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')
# # load previous models, both for (continuous) training and validation (inference)
# flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + '1493886629',
#                     'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')

# important to provide acess to random seed, to reproduce the experiment
flags.DEFINE_integer('random_seed', 7, 'Value of random seed')      # e.g. random.randint(0, sys.maxsize)


def main(_):
    config = flags.FLAGS.__flags.copy()
    # use json to pass fixed_params to the shell
    config['fixed_params'] = json.loads(config['fixed_params'])

    if config['fullsearch']:
        # some code for HP search
        print('HP search is under construction')
        pass
    else:
        model = make_model(config)

        if config['infer']:
            # code for testing/inferring
            model.infer()
        else:
            # code for training
            model.train()

if __name__ == '__main__':
    tf.app.run()
