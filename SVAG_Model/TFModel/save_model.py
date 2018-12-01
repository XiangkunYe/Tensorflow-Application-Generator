import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


def save_model(output_dir, output_name, model_file):
    # clear and reset session
    # K.clear_session()
    sess = tf.Session(graph=tf.Graph())
    K.set_session(sess)
    K.set_learning_phase(0)
    # load hdf5 model file
    model = load_model(model_file)
    # get graph
    gd = sess.graph.as_graph_def()

    constant_graph = graph_util.convert_variables_to_constants(sess, gd, ['final_output/Softmax'])
    graph_io.write_graph(constant_graph, output_dir, output_name, as_text=False)
