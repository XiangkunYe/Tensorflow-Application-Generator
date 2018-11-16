from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


def save_model(model, output_dir, output_name):
    orig_output_node_names = [node.op.name for node in model.outputs]
    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               orig_output_node_names)

    graph_io.write_graph(constant_graph, str(output_dir), output_name,
                         as_text=False)
