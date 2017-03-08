import tensorflow as tf
# Note tf.contrib.keras from mid-March 2017, TF v1.1
#   and tf.keras by TF v1.2
# At the current stage, "AttributeError: 'module' object has no attribute 'keras'"

video = tf.keras.layers.Input(shape=(None, 150, 150, 3))
cnn = tf.keras.application.InceptionV3(weights='imagenet',
                                       include_top=False,
                                       pool='avg')
cnn.trainable = False
encoded_frames = tf.keras.layers.TimeDistributed(cnn)(video)
encoded_vid = tf.layers.LSTM(256)(encoded_frames)

question = tf.keras.layers.Input(shape=(100), dtype='int32')
x = tf.keras.layers.Embedding(10000, 256, mask_zero=True)(question)
encoded_q = tf.keras.layers.LSTM(128)(x)

x = tf.keras.layers.concat([encoded_vid, encoded_q])
x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(1000)(x)

model = tf.keras.models.Model([video, question], outputs)
model.compile(optimizer=tf.AdamOptimizer(),
              loss=tf.softmax_crossentropy_with_logits)
