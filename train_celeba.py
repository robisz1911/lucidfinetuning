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

from tensorflow.core.protobuf import saver_pb2

import sys
import dataset
import freeze_graph
import argparse
import os
import matplotlib.pyplot as plt

import pandas as pd

from keras_preprocessing.image import ImageDataGenerator

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
parser.add_argument("--batch_size", help='batch size', default=32, type=int)
parser.add_argument("--nb_epoch", help='number of epochs', default=200, type=int)
parser.add_argument("--dataset", help='training dataset', default="flowers17", type=str)
parser.add_argument("--cutoff", help='freeze the first cutoff layers during training', default=0, type=int)
parser.add_argument("--do_load_model", help='True or False', default=False, type=str2bool)
parser.add_argument("--do_save_model", help='True or False', default=False, type=str2bool)
parser.add_argument("--do_finetune", help='True or False', default=True, type=str2bool)
parser.add_argument("--save_layer_names_shapes", help='if yes : iterate over act layers and return name/shape', default=False, type=str2bool)
parser.add_argument("--weights", help='default -> imagenet weights, None -> random initialized', default='imagenet')
parser.add_argument("--top_node", help='top layer to visualize', default='Mixed_5c_Concatenated/concat', type=str)


FLAGS = parser.parse_args()

batch_size = FLAGS.batch_size
nb_epoch = FLAGS.nb_epoch
data = FLAGS.dataset
cutoff = FLAGS.cutoff
do_load_model = FLAGS.do_load_model
do_save_model = FLAGS.do_save_model
do_finetune = FLAGS.do_finetune
weights = FLAGS.weights
top_layer = FLAGS.top_node
save_layer_names_shapes = FLAGS.save_layer_names_shapes

if weights=="None":
    weights=None

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
    output_node_names = top_node # default: "Mixed_5c_Concatenated/concat"
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
    nb_classes = tags
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if do_load_model == True:
        model.load_weights('my_model_weights.h5')

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

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

    model.compile(optimizer='adadelta', loss='cosine_proximity', metrics=['binary_accuracy'])
    model.fit_generator(train_flow, steps_per_epoch=(train_samples_per_epoch//batch_size), nb_epoch=nb_epoch,
        validation_data=test_flow, validation_steps=test_samples_per_epoch//batch_size, callbacks=[history])

    if do_save_model:
        model.save_weights('my_model_weights.h5')

class CelebA():
    '''Wraps the celebA dataset, allowing an easy way to:
         - Select the features of interest,
         - Split the dataset into 'training', 'test' or 'validation' partition.
    '''
    def __init__(self, main_folder='celeba-dataset/', selected_features=None, drop_features=[]):
        self.main_folder = main_folder
        self.images_folder   = os.path.join(main_folder, 'img_align_celeba/img_align_celeba/')
        self.attributes_path = os.path.join(main_folder, 'list_attr_celeba.csv')
        self.partition_path  = os.path.join(main_folder, 'list_eval_partition.csv')
        self.selected_features = selected_features
        self.features_name = []
        self.__prepare(drop_features)

    def __prepare(self, drop_features):
        '''do some preprocessing before using the data: e.g. feature selection'''
        # attributes:
        if self.selected_features is None:
            self.attributes = pd.read_csv(self.attributes_path)
            self.num_features = 40
        else:
            self.num_features = len(self.selected_features)
            self.selected_features = self.selected_features.copy()
            self.selected_features.append('image_id')
            self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

        # remove unwanted features:
        for feature in drop_features:
            if feature in self.attributes:
                self.attributes = self.attributes.drop(feature, axis=1)
                self.num_features -= 1
      
        self.attributes.set_index('image_id', inplace=True)
        self.attributes.replace(to_replace=-1, value=0, inplace=True)
        self.attributes['image_id'] = list(self.attributes.index)
  
        self.features_name = list(self.attributes.columns)[:-1]
  
        # load ideal partitioning:
        self.partition = pd.read_csv(self.partition_path)
        self.partition.set_index('image_id', inplace=True)
  
    def split(self, name='training', drop_zero=False):
        '''Returns the ['training', 'validation', 'test'] split of the dataset'''
        # select partition split:
        if name is 'training':
            to_drop = self.partition.where(lambda x: x != 0).dropna()
        elif name is 'validation':
            to_drop = self.partition.where(lambda x: x != 1).dropna()
        elif name is 'test':  # test
            to_drop = self.partition.where(lambda x: x != 2).dropna()
        else:
            raise ValueError('CelebA.split() => `name` must be one of [training, validation, test]')

        partition = self.partition.drop(index=to_drop.index)
      
        # join attributes with selected partition:
        joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

        if drop_zero is True:
            # select rows with all zeros values
            return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
        elif 0 <= drop_zero <= 1:
            zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
            zero = zero.sample(frac=drop_zero)
            return joint.drop(index=zero.index)

        return joint



#def load_data():
#    X, y, tags = dataset.dataset(data, 299)
#    prinleba.attributes.sample(5))
#
#    nb_classes = len(tags)
#
#    sample_count = len(y)
#    train_size = sample_count * 4 // 5
#    X_train = X[:train_size]
#    y_train = y[:train_size]
#    Y_train = to_categorical(y_train, nb_classes)
#    X_test  = X[train_size:]
#    y_test  = y[train_size:]
#    Y_test = to_categorical(y_test, nb_classes)
#
#    datagen = ImageDataGenerator(
#        featurewise_center=False,
#        samplewise_center=False,
#        featurewise_std_normalization=False,
#        samplewise_std_normalization=False,
#        zca_whitening=False,
#        rotation_range=0,
#        width_shift_range=0.125,
#        height_shift_range=0.125,
#        horizontal_flip=True,
#        vertical_flip=False,
#        fill_mode='nearest')
#
#    datagen.fit(X_train)
#    train_flow = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
#    test_flow = datagen.flow(X_test, Y_test, batch_size=batch_size)
#    train_samples_per_epoch = X_train.shape[0]
#    test_samples_per_epoch = X_test.shape[0]
#    return train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_e
def main():
    topology = "googlenet"
    if topology == "googlenet":
        net = InceptionV1(include_top=False, weights=weights, input_tensor=None, input_shape=(224, 224, 3), pooling=None)
        top_node = top_layer
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
#        celeba = CelebA(drop_features=[])
#        print(celeba.attributes.sample(5))
#        celeba = CelebA(drop_features=['Attractive','Pale_Skin','Blurry',])

#        train_flow, test_flow, tags, train_samples_per_epoch, test_samples_per_epoch, celeba = load_data()
#        print(celeba.attributes.sample(5))
        celeba = CelebA(drop_features=[""])
        #data = celeba.values
        #index = celeba.features_name
        #print(data)
        #print(index)
        

        # auigumentations for training set:
        train_datagen = ImageDataGenerator(rotation_range=20,
                                           rescale=1./255,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
        
        # only rescaling the validation set
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        # get training and validation set:
        train_split = celeba.split('training'  , drop_zero=True)
        valid_split = celeba.split('validation', drop_zero=True)

        train_indexes = list(train_split['image_id'])
        valid_indexes = list(valid_split['image_id'])

        print(len(train_indexes))

        print(len(valid_indexes))

        print(len(celeba.features_name))
        print(len(train_indexes))
        print(len(valid_indexes))

        train_data = train_split.values[:,0:41]
        valid_data = valid_split.values[:,0:41]

        print(train_data)
        print(valid_data)

        celeba.features_name.append('image_id')
        print(celeba.features_name[0:40])
        print(celeba.features_name)

        df_train = pd.DataFrame(train_data, columns = celeba.features_name)
        df_valid = pd.DataFrame(valid_data, columns = celeba.features_name)
        
        cols=[i for i in df_train.columns if i not in ["image_id"]]
        for col in cols:
            df_train[col] = pd.to_numeric(df_train[col])
            df_valid[col] = pd.to_numeric(df_valid[col])
        #print(df_train)
        print(celeba.features_name)
        print(":::")
        print(df_train['image_id'].values)

        #print("split, done")
        #print(train_split.loc[0,'Bald'])
        #print(train_split.values)
        # data generators:
        #print(batch_size)
        #print(celeba.features_name)
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            directory=celeba.images_folder,
            x_col='image_id',
            y_col=celeba.features_name[:40],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='other'
        )
        print("train_generator done")
        print(train_generator)
        valid_generator = valid_datagen.flow_from_dataframe(
            dataframe=df_valid,
            directory=celeba.images_folder,
            x_col='image_id',
            y_col=celeba.features_name[:40],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='other'
        )
        print("teszt:")
        print(len(train_generator))

        finetune(net, train_generator, valid_generator, 40, len(train_generator), len(valid_generator))

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
            print(layer.name, layer.output_shape)
            if "Concat" in layer.name:
                text_file.write(str(layer.name)+"/concat "+str(layer.get_output_at(0).get_shape().as_list()[3])+"\n")
        text_file.close()
        exit()


if __name__ == "__main__":
    main()



