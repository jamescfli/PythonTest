from rnn_by_tf_3_cuscell_dropout import tf, reset_graph, CustomCell, vocab_size, np, vocab_to_idx, idx_to_vocab, time, train_network
from rnn_by_tf_4_layernorm import LayerNormalizedLSTMCell


def build_graph(
    cell_type = None,
    num_weights_for_custom_cell = 5,
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    build_with_dropout=False,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    dropout = tf.constant(1.0)

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    if cell_type == 'Custom':
        cell = CustomCell(state_size, num_weights_for_custom_cell)
    elif cell_type == 'GRU':
        cell = tf.contrib.rnn.core_rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.contrib.rnn.core_rnn_cell.BasicRNNCell(state_size)

    if build_with_dropout:
        cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

    if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
        cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers)

    if build_with_dropout:
        cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    # reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x = x,
                y = y,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step,
                preds = predictions,
                saver = tf.train.Saver())


def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict = {g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict = {g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'], g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0  # put less significant values to 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]  # randomly choose from top chars
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)
    chars = map(lambda x: idx_to_vocab[x], chars)
    print("".join(chars))
    return "".join(chars)

g = build_graph(cell_type='GRU', num_steps=1, batch_size=1)
generate_characters(g, "saves/GRU_20_epochs", 750, prompt='A', pick_top_chars=5)

# # new data to learn
# file_name = 'variousscripts.txt'

import urllib                       # python2
# import urllib.request as urllib     # py3


def web_lookup(url, saved={}):
    if url in saved:
        return saved[url]
    page = urllib.urlopen(url).read()
    saved[url] = page
    return page

file_url = 'https://gist.githubusercontent.com/spitis/59bfafe6966bfe60cc206ffbb760269f/' + \
    'raw/030a08754aada17cef14eed6fac7797cda830fe8/variousscripts.txt'
# import os
# print os.getcwd()
file_name = './saves/variousscripts.txt'
file_content = web_lookup(file_url)
print("Data length:", len(file_content))
with open(file_name, 'w') as f:
    f.write(file_content)
    print('File content was written to ' + file_name)

with open(file_name,'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

vocab = set(raw_data)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

data = [vocab_to_idx[c] for c in raw_data]
del raw_data

g = build_graph(cell_type='GRU',
                num_steps=80,
                state_size = 512,
                batch_size = 50,
                num_classes=vocab_size,
                learning_rate=5e-4)
t = time.time()
losses = train_network(g, 30, num_steps=80, batch_size = 50, save="saves/GRU_30_epochs_variousscripts")
print("It took", time.time() - t, "seconds to train for 30 epochs.")    # loss: 0.64769 in 18697 secs
print("The average loss on the final epoch was:", losses[-1])

g = build_graph(cell_type='GRU', num_steps=1, batch_size=1, num_classes=vocab_size, state_size = 512)
generate_characters(g, "saves/GRU_30_epochs_variousscripts", 750, prompt='D', pick_top_chars=5)

# # The script does not make any sense to me.
# D?;EB7I-
# :Pi jXUhU- iYh! MXPj SedjUdji Yi SXYbU3
#
# 76M2H60
# IYh- P mecPdT
# :PjX PffPhUT XUhU- XYUjX- Ro jXeiU jme
# TPdiYiX,
#
# 8Yhij 5YjYqUd0
# ;V jXYi- ; iPo- ; fhPo- Veh mXehU!
# FbUPiU ; SPd- FUjhkSXYe- HYSXPhT!
#
# FHEIF7HE0
#
# F2KB;D20
# :U iXPbb dej- iYh- ; TPhU jPbZYdW((
#
# Fheleij0
# ;'bb dej WeeT VehjX1 ; ZdUm XYc e'Uh PdT ed'j,
#
# 6K?7 L;D57DJ;E0
# 5PjUiRo1 ; Pc hPhdU je RU ckSX,
#
# F7JHK5:;E0
# FbPdjPWUdUj! BUj ki je jXU Skijec-
# :YTUi oek dem- PdT We ed,
#
# F7JHK5:;E0
# MXo- jXUd XU iXPbb jXek XPTi1 Rkj jXPj jXek mUbSecU,
#
# 52FKB7J0
# 9eT iXPbb RU mUYdj
# Je jPZU jXU WbTUjYed eV jXo iUSehT SPbb-
# :PlU ijYVj kfed jXo ied',
#
# L;H9;B;20
# MXUd co ied'i cUPdi cPZU ThPVjUT WedU- co behT-
# 5ecU XYciUbV- jPZU XYc- Vhec jXek XebT je XYi feYdj-
# 2dT ifYj Yj dem- PdT ;'bb ijhPYWXj Se
