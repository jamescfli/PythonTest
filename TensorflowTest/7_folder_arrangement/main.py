# The shell APIs
# can be further split into train.py and infer.py, but two APIs are more preferred
import os
import json
import random
import sys
import tensorflow as tf


from models import make_models      # helper function to load any models you have
from hpsearch import hyperband, randomsearch

# make my paths absolute to be independent from where python binary is called
dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

# HP search config
flags.DEFINE_boolean('fullsearch', False,
                     'Perform a full search of HP space ex:(hyperband -> lr search -> hyperband with best lr)')
flags.DEFINE_boolean('dry_run', False, 'Perform a dry_run (testing purpose)')
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform HP search')

# fix some HP inside model, e.g. explore diff models fixing the lr
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a HP search, ex \'{"lr": 0.001}\'')

# agent config
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('max_iter', 2000, 'Number of training steps')
flags.DEFINE_boolean('infer', False, 'Load an agent for playing')

# important for TensorBoard
# choose to name the output folder
flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + str(int(time.time())),
                    'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')

# important to provide acess to random seed, to reproduce the experiment
flags.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Vlaue of random seed')

def main(_):
    config = flags.FLAGS.__flags.copy()
    # use json to pass fixed_params to the shell
    config['fixed_params'] = json.loads(config['fixed_params'])

    if config['fullsearch']:
        # some code for HP search
        pass
    else:
        model = make_models(config)

        if config['infer']:
            # code for infer
            pass
        else:
            # code for training
            pass

if __name__ == '__main__':
    tf.app.run()
