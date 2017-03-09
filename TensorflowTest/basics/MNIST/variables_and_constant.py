import tensorflow as tf


const = tf.constant(1.0, name='constant')
print(tf.get_default_graph().as_graph_def())
# node {
#   name: "constant"
#   op: "Const"
#   attr {
#     key: "dtype"
#     value {
#       type: DT_FLOAT
#     }
#   }
#   attr {
#     key: "value"
#     value {
#       tensor {
#         dtype: DT_FLOAT
#         tensor_shape {
#         }
#         float_val: 1.0
#       }
#     }
#   }
# }
# versions {
#   producer: 21
# }
# .. where constant is within the graph
# .. and it is mem consuming due to buried in the network itself

with tf.Session() as sess:
    print(const.eval())     # 1.0
