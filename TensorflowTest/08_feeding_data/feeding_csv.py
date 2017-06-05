import tensorflow as tf

filename_queue = tf.train.string_input_producer(['csv/file0.csv', 'csv/file1.csv'])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)    # each read reads a single line from the file

# print key       # Tensor("ReaderReadV2:0", shape=(), dtype=string)
# print value     # Tensor("ReaderReadV2:1", shape=(), dtype=string)

record_defaults = [[10], [10], [10], [10], [10]]
# Convert CSV records to tensors. Each column maps to one tensor
# follow rfc4180 format:
#   aaa,bbb,ccc CRLF
#   zzz,yyy,xxx
col1, col2, col3, col4, col5 = tf.decode_csv(value,
                                             record_defaults=record_defaults,
                                             field_delim=',')
print(col1.__class__)
features = tf.stack([col1, col2, col3, col4])


with tf.Session() as sess:
    # # vanilla test, does not work
    # labels = sess.run([col5])
    # print labels
    # # .. must call tf.train.start_queue_runners to populate the queue before you call run
    # # .. otherwise read will block while it waits for filenames from the queue

    # coordinate the termination of a set of threads
    coord = tf.train.Coordinator()  # A coordinator for threads
    # starts all queue runners collected in the graph, return: a list of threads
    threads = tf.train.start_queue_runners(coord=coord)
    # print threads   # 2 threads on MBP

    for i in range(12):
        # start a thread
        example, label = sess.run([features, col5])
        print example
        print label

    print sess.run([col1])      # [1]
    print sess.run([col1])      # [2]
    print sess.run([col1])      # [3]
    print sess.run([col1])      # [4]
    print sess.run([col1])      # [1] ..

    coord.request_stop()
    coord.join(threads)     # Wait for all the threads to terminate
