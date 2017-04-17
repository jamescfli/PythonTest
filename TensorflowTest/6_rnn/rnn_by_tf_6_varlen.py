from __future__ import print_function
import pandas as pd, numpy as np, tensorflow as tf
import blogs_data   # available at https://github.com/spitis/blogs_data

# classify multiple sentences of different lengths at the same time


df = blogs_data.loadBlogs().sample(frac=1).reset_index(drop=True)
vocab, reverse_vocab = blogs_data.loadVocab()
train_len, test_len = np.floor(len(df)*0.8), np.floor(len(df)*0.2)
train, test = df.ix[:train_len-1], df.ix[train_len:train_len+test_len]
df = None   # del df
train.head()


# construct iterator
class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n
        return res['as_numbers'], res['gender']*3 + res['age_bracket'], res['length']

data = SimpleDataIterator(train)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')
print('Target values\n', d[1], end='\n\n')
print('Sequence lengths\n', d[2])


class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n

        # pad sequences with 0s to meet the max_len
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_number'].values[i]
        return x, res['gender']*3 + res['age_bracket'], res['length']

data = PaddedDataIterator(train)
d = data.next_batch(3)
# now returns a single input matrix of dimension [batch_size, max_sequence_length]
print('Input sequences\n', d[0], end='\n\n')


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def build_graph(vocab_size = len(vocab),
                state_size = 64,
                batch_size = 256,
                num_classes = 6):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, None])    # variable 'steps'
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(1.0)    # no dropout if 1.0

    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.contrib.rnn.core_rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size],
                                 initializer=tf.constant_initializer(0.0))
    # creates a new tensor by replicating input multiples times
    # e.g. tiling [a b c d] by [2] produces [a b c d a b c d]
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)

    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    """
    Obtain the last relevant output. The best approach in the future will be to use:

        last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

    which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
    gradient for this op has not been implemented as of this writing.

    The below solution works, but throws a UserWarning re: the gradient.
    """
    # # Option 1:
    # last_rnn_output = tf.gather_nd(rnn_outputs, tf._pack([tf.range(batch_size), seqlen-1], axis=1))
    # Option 2:
    idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen-1)
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {'x': x,
            'seqlen': seqlen,
            'y': y,
            'dropout': keep_prob,
            'loss': loss,
            'ts': train_step,
            'preds': preds,
            'accuracy': accuracy}


def train_graph(graph, batch_size=256, num_epochs=10, iterator=PaddedDataIterator):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tr = iterator(train)
        te = iterator(test)

        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_
                te_losses.append(accuracy / step)
                step, accuracy = 0, 0
                print("Accuracy after epoch", current_epoch, '- tr:', tr_losses[-1], '- te', te_losses[-1])
    return tr_losses, te_losses

g = build_graph()
tr_losses, te_losses = train_graph(g)


class BucketDataIterator():
    def __init__(self, df, num_buckets = 5):
        df = df.sort_values('length').reset_index(drop=True)
        self.size = len(df) / num_buckets
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.ix[bucket*self.size: (bucket+1)*self.size - 1])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor+n+1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0,self.num_buckets)

        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i]+n-1]
        self.cursor[i] += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['gender']*3 + res['age_bracket'], res['length']

from time import time
g = build_graph()
t = time()
tr_losses, te_losses = train_graph(g, num_epochs=1, iterator=PaddedDataIterator)
print("Total time for 1 epoch with PaddedDataIterator:", time() - t)
