import os
import copy     # shallow and deep copy
import json
import tensorflow as tf


class BasicAgent(object):
    # build model, need to pass a 'configuration' dictionary
    def __init__(self, config):
        # keep the best HP so far inside the model
        if config['best']:
            config.update(self.get_best_config(config['env_name']))

        # make deep copy before any potential mutation over configurations
        self.config = copy.deepcopy(config)

        if config['debug']:
            print('config', self.config)

        # remember to store random seed and use it in tf.set_random_seed()
        self.random_seed = self.config['random_seed']

        # copy shared basic HP to the model
        self.result_dir = self.config['result_dir']
        self.max_iter = self.config['max_iter']
        self.lr = self.config['lr']
        self.nb_units = self.config['nb_units']
        # .. and etc

        # customize child models without inheritance hell, override function completely
        self.set_agent_props()

        self.graph = tf.get_default_graph()

        # any operations that should be in the graph can be added in this way
        # this line overrides the current default graph for the lifetime of the context
        with self.graph.as_default():
            # the following line within graph.as_default() makes saver record all tensors
            self.graph = self.build_graph(self.graph)
            self.saver = tf.train.Saver(
                max_to_keep=50,
                # write_version=tf.train.SaverDef.V2,     # default already
            )

        # other common code for initialization
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        # prepare tf summary
        self.tf_summary()

        # this is not always common to all models, so it is again separated from __init__ one
        # parameter initialization is done in here
        self.init()
        # your model is expected to be ready now

    def set_agent_props(self):
        # this is here to be overriden completely
        # to know exactly which options it needs
        pass

    def get_best_config(self):
        # return dictionary used to update the initial config
        pass

    @staticmethod
    def get_random_config(fixed_params={}):
        # static, because you want to be able to pass this to other processes
        # so they can independently generate random config of the current model
        raise Exception('The get_random_config function must be overriden by the agent')

    def build_graph(self):
        raise Exception('The build_graph function must be overriden by the agent')

    def infer(self):
        raise Exception('The infer function must be overriden by the agent')

    def learn_from_epoch(self):
        # separate func to train per epoch and func to train globally
        raise Exception('The learn_from_epoch function must be overriden by the agent')

    def tf_summary(self):
        raise Exception('The tf_summary function must be overriden by the agent')
        # basic form: self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)

    def train(self):
        # usually common to all models
        for epoch_id in range(0, self.max_iter):
            if self.config['debug']:
                print "Epoch:", '%04d' % (epoch_id+1)
            self.learn_from_epoch()

            if self.config['save_every'] > 0 and (epoch_id+1) % self.config['save_every'] == 0:
                self.save()
        if self.config['debug']:
            print "optimization finished."

    def save(self):
        global_step = tf.train.global_step(self.sess, self.global_step_t)

        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/agent', global_step)

        # always keep config of that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # usually common to all models
        # separation makes overriden clean
        # this is an example
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            # # initialize randomly by init_op
            # self.sess.run(self.init_op)
            # or
            self.sess.run(tf.global_variables_initializer())
        else:
            # load the last saved model in the existing folder
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

