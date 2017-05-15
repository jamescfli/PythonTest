from __future__ import print_function

import tensorflow as tf
from google.protobuf.json_format import MessageToJson

PATH = '/tmp/data/'
FILE = 'valid.tfrecords'

for example in tf.python_io.tf_record_iterator(PATH+FILE):
    result = tf.train.Example.FromString(example)
    json_msg = MessageToJson(tf.train.Example.FromString(example))
    print(json_msg)

# features -> feature -> image_raw -> byteList -> value -> ***
#                     -> depth    -> int64List -> value -> 1
#                     -> label
#                     -> width
#                     -> height

# json_msg{
#   "features": {
#     "feature": {
#       "image_raw": {
#         "bytesList": {
#           "value": [
#             "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADEhISfoivGqb/938AAAAAAAAAAAAAAAAeJF6aqv39/f394az98sNAAAAAAAAAAAAAAAAx7v39/f39/f39+11SUjgnAAAAAAAAAAAAAAAAEtv9/f39/ca29/EAAAAAAAAAAAAAAAAAAAAAAABQnGv9/c0LACuaAAAAAAAAAAAAAAAAAAAAAAAAAA4Bmv1aAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIv9vgIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALvv1GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACPx4aBsAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUfD9/XcZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAtuv39lhsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBd/P27AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPn9+UAAAAAAAAAAAAAAAAAAAAAAAAAAAAAugrf9/c8CAAAAAAAAAAAAAAAAAAAAAAAAACeU5f39/fq2AAAAAAAAAAAAAAAAAAAAAAAAGHLd/f39/clOAAAAAAAAAAAAAAAAAAAAAAAXQtX9/f39xlECAAAAAAAAAAAAAAAAAAAAABKr2/39/f3DUAkAAAAAAAAAAAAAAAAAAAAAN6zi/f39/fSFCwAAAAAAAAAAAAAAAAAAAAAAAIj9/f3Uh4QQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
#           ]
#         }
#       },
#       "depth": {
#         "int64List": {
#           "value": [
#             "1"
#           ]
#         }
#       },
#       "label": {
#         "int64List": {
#           "value": [
#             "5"
#           ]
#         }
#       },
#       "width": {
#         "int64List": {
#           "value": [
#             "28"
#           ]
#         }
#       },
#       "height": {
#         "int64List": {
#           "value": [
#             "28"
#           ]
#         }
#       }
#     }
#   }
# }