import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda

def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice no. [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    tensor_shape = K.shape(x)
    nb_slices = tensor_shape[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*nb_slices:]   # the rest for the batch
    return x[part*nb_slices:(part+1)*nb_slices]

def to_multi_gpu(model, n_gpus=2):
    """
    Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.
    """
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:'+str(g)):
            # get the g-th slice of the batch
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)
            # apply model on the slice
            towers.append(model(slice_g))
    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    return Model(input=[x], output=merged)


if __name__ == '__main__':
    # build your test model
    from keras.layers.convolutional import Convolution2D
    from keras.layers.core import Activation
    import numpy as np

    def get_model():
        input_tensor = Input((96,96,1), name="input1")
        output = Convolution2D(64, 5, 5, border_mode='same', name="conv1")(input_tensor)
        output = Activation('relu', name="relu1")(output)
        # more layers
        model = Model(input=input_tensor, output=output)
        model.compile(optimizer='rmsprop', loss='mse')
        return model

    model = get_model()
    model = to_multi_gpu(model)
    x = np.random.rand(1000, 96, 96, 1)
    y = model.predict(x, verbose=True)
    # or model.fit()
