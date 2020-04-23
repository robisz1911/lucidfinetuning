import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
import time

from keras.utils import to_categorical
from tensorflow.core.protobuf import saver_pb2
import freeze_graph
import argparse
import os
from tensorflow.python.saved_model import tag_constants


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help='epoch', type=int)

FLAGS = parser.parse_args()

epoch = FLAGS.epoch


def create_final_pb_files(graph_file, ckpt_file, top_node, frozen_model_file):
    input_graph = graph_file
    input_saver = ""
    input_binary = True
    input_checkpoint = ckpt_file
    output_node_names = top_node # "Mixed_5c_Concatenated/concat"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph = frozen_model_file
    clear_devices = True
    initializer_nodes = ""
    variable_names_whitelist = ""
    variable_names_blacklist = ""
    input_meta_graph = ""
    input_saved_model_dir = ""
    #from tensorflow.python.saved_model import tag_constants
    saved_model_tags = tag_constants.SERVING
    checkpoint_version = saver_pb2.SaverDef.V2
    freeze_graph.freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_whitelist,
                 variable_names_blacklist,
                 input_meta_graph,
                 input_saved_model_dir,
                 saved_model_tags,
                 checkpoint_version)

def main():
    graph_file = "model.pb"
    ckpt_file = "model.ckpt"
    frozen_model_file = ".pb"
    top_node = "Mixed_5c_Concatenated/concat"
    create_final_pb_files(graph_file, str(epoch)+ckpt_file, top_node, str(epoch)+frozen_model_file)

if __name__ == "__main__":
    main()

