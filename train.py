import numpy as np
import tensorflow as tf
import keras.backend as K
import keras

from keras.utils import to_categorical
from keras.optimizers import SGD, Adam, RMSprop

# Import necessary components to build AlexNet
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from inception_v1 import InceptionV1

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.core.protobuf import saver_pb2

import sys
import dataset
import freeze_graph
import argparse
import os
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help='batch size', type=int)
parser.add_argument("--nb_epoch", help='number of epochs', type=int)
parser.add_argument("--dataset", help='training dataset', type=str)
parser.add_argument("--cutoff", help='freeze the first cutoff layers during training', type=int)
parser.add_argument("--do_load_model", help='True or False', type=str2bool)
parser.add_argument("--do_save_model", help='True or False', type=str2bool)
parser.add_argument("--do_finetune", help='True or False', type=str2bool)
parser.add_argument("--save_layer_names_shapes", help='if yes : iterate over act layers and return name/shape', default=False, type=str2bool)

FLAGS = parser.parse_args()

batch_size = FLAGS.batch_size
nb_epoch = FLAGS.nb_epoch
data = FLAGS.dataset
cutoff = FLAGS.cutoff
do_load_model = FLAGS.do_load_model
do_save_model = FLAGS.do_save_model
do_finetune = FLAGS.do_finetune

save_layer_names_shapes = FLAGS.save_layer_names_shapes


def save(graph_file, ckpt_file, top_node, frozen_model_file):
    sess = K.get_session()
    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
    save_path = saver.save(sess, ckpt_file)

    gd = sess.graph.as_graph_def()
    tf.train.write_graph(gd, ".", graph_file, False)

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
    from tensorflow.python.saved_model import tag_constants
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

# create callbacks for accuracy tracking
class AccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.valacc = []
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.valacc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))


def finetune(base_model, train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch):
    nb_classes = len(tags)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if do_load_model == True:
        model.load_weights('my_model_weights.h5')
        print("weights loaded")

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    #for i, layer in enumerate(base_model.layers):
    #    print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit_generator(train_flow, steps_per_epoch=train_samples_per_epoch//batch_size, nb_epoch=0,
    #     validation_data=test_flow, validation_steps=test_samples_per_epoch//batch_size)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 'cutoff' layers and unfreeze the rest. Francois used 249
    
    print(cutoff)
    # cutoff = 20
    for layer in model.layers[:cutoff]:
        layer.trainable = False
    for layer in model.layers[cutoff:]:
        layer.trainable = True

    model.summary()
    
    history=AccHistory()	

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_flow, steps_per_epoch=train_samples_per_epoch//batch_size, nb_epoch=nb_epoch,
        validation_data=test_flow, validation_steps=test_samples_per_epoch//batch_size, callbacks=[history])
    
    if do_save_model:
        model.save_weights('my_model_weights.h5')
        print("weights saved")


    print(history.acc)
    print(history.valacc)

def load_data():
    X, y, tags = dataset.dataset(data, 299)
    print(tags)

    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count * 4 // 5
    X_train = X[:train_size]
    y_train = y[:train_size]
    Y_train = to_categorical(y_train, nb_classes)
    X_test  = X[train_size:]
    y_test  = y[train_size:]
    Y_test = to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    datagen.fit(X_train)
    train_flow = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
    test_flow = datagen.flow(X_test, Y_test, batch_size=batch_size)
    train_samples_per_epoch = X_train.shape[0]
    test_samples_per_epoch = X_test.shape[0]
    return train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch

def main():
    topology = "googlenet"
    if topology == "googlenet":
        net = InceptionV1(include_top=False, weights=None, input_tensor=None, input_shape=(299, 299, 3), pooling=None)
        top_node = "Mixed_5c_Concatenated/concat"
        frozen_model_file = "googlenetLucid.pb"
    elif topology == "inception_v3":
        net = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)
        top_node = "mixed10/concat"
        frozen_model_file = "inceptionv3Lucid.pb"
    elif topology == "vgg16":
        net = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)
        top_node = "block5_pool/MaxPool"
        frozen_model_file = "vgg16Lucid.pb"
    else:
        assert False, "unknown topology " + topology


    if do_finetune:
        train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch = load_data()
        finetune(net, train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch)

    graph_file = "model.pb"
    ckpt_file = "model.ckpt"

    print("saving", graph_file, ckpt_file, frozen_model_file, "with top node", top_node)
    save(graph_file, ckpt_file, top_node, frozen_model_file)

    do_dump = False
    if do_dump:
        graph_def = tf.GraphDef()
        with open(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print(node.name)

    save_layer_names_shapes=False
    if save_layer_names_shapes:
        text_file = open("layers_name_channel_concat.txt", "w")

        for layer in model.layers:
            #print(layer.name, layer.output_shape)
            if "Concat" in layer.name:
                text_file.write(str(layer.name)+"/concat "+str(layer.get_output_at(0).get_shape().as_list()[3])+"\n")
        text_file.close()
        exit()


if __name__ == "__main__":
    main()



