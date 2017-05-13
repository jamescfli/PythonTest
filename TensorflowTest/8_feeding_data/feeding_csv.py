import tensorflow as tf

filename_queue = tf.train.string_input_producer(['csv/file0.csv', 'csv/file1.csv'])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

print key       # Tensor("ReaderReadV2:0", shape=(), dtype=string)
print value     # Tensor("ReaderReadV2:1", shape=(), dtype=string)

record_defaults = [[10], [10], [10], [10], [10]]
# pa
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(12):
        example, label = sess.run([features, col5])
        print example
        print label

    coord.request_stop()
    coord.join(threads)
