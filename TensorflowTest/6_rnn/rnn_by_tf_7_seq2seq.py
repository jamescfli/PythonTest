from __future__ import print_function
import pandas as pd, numpy as np, tensorflow as tf


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def build_seq2seq_graph(vocab_size = len(vocab),
                        state_size = 64,
                        batch_size = 256,
                        num_classes = 6):
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, None])
    seqlen = tf.placeholder(tf.int32, [batch_size])     # with int number for the sequences in the batch
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(1.0)

    y_ = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(x)[1]])     # batch_size, num_steps

    """
    Create a mask that we will use for the cost function

    This mask is the same shape as x and y_, and is equal to 1 for all non-PAD time
    steps (where a prediction is made), and 0 for all PAD time steps (no pred -> no loss)
    The number 30, used when creating the lower_triangle_ones matrix, is the maximum
    sequence length in our dataset
    """
    lower_triangular_ones = tf.constant(np.tril(np.ones([30, 30])), dtype=tf.float32)
    seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, seqlen - 1), [0,0], [batch_size, tf.reduce_max(seqlen)])

    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.contrib.rnn.core_rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size], initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen,
                                                 initial_state=init_state)

    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y_, [-1])

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(rnn_outputs, W) + b

    preds = tf.nn.softmax(logits)

    correct = tf.cast(tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y_reshaped),tf.int32) * \
              tf.cast(tf.reshape(seqlen_mask, [-1]),tf.int32)

    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(seqlen, tf.float32))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped)
    loss = loss * tf.reshape(seqlen_mask, [-1])

    loss = tf.reduce_sum(loss) / tf.reduce_sum(seqlen_mask)

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


class BucketedDataIterator():
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


g = build_seq2seq_graph()
tr_losses, te_losses = train_graph(g, iterator=BucketedDataIterator)
