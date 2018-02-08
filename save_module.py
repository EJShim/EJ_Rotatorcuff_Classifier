import os
import tensorflow as tf
import network.VRN_64_TF as config_module

weight_path = os.path.join("./", "weights", "2block", "epoch49model.ckpt")

tensor_in, y, keep_prob, last_conv = config_module.get_very_shallow_model()
tf.add_to_collection("input_tensor", tensor_in)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("output_tensor", y)
tf.add_to_collection("last_conv", last_conv)

sess = tf.Session()
#Restore Graph
try:
    saver = tf.train.Saver()
    saver.restore(sess, weight_path)
except Exception as e:
    print(e)
    sess.run(tf.global_variables_initializer())




builder = tf.saved_model.builder.SavedModelBuilder("./weights/2block_49")
builder.add_meta_graph_and_variables(sess, ['foo-tag'])
builder.save()
