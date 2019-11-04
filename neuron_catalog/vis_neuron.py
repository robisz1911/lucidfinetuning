import sys
import numpy as np
import scipy.misc
from lucid.modelzoo.vision_base import Model
import lucid.optvis.render as render
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_PATH", help='model path')
parser.add_argument("--LAYER", help='layer name')
parser.add_argument("--NEURON_INDEX", help='neuron index', type=int)

FLAGS = parser.parse_args()

MODEL_PATH=FLAGS.MODEL_PATH
LAYER=FLAGS.LAYER
NEURON_INDEX=FLAGS.NEURON_INDEX

class FrozenNetwork(Model):
    model_path = MODEL_PATH
    image_shape = [299, 299, 3]
    image_value_range = (0, 1)
    input_name = 'input_1'


network = FrozenNetwork()
network.load_graphdef()


#for layer in network:
#    print(layer.get_shape())

pixels = 256

param_f = lambda: param.image(pixels, fft=True, decorrelate=True)
#obj_test = objectives.channel(LAYER, NEURON_INDEX).get_shape()
#print(obj_test)
obj = objectives.channel(LAYER, NEURON_INDEX)
images = render.render_vis(network, obj, param_f, thresholds=(2048,))
assert len(images)==1
image = images[0]
assert len(image)==1
image = image[0]

out_filename = LAYER.replace("/","-") + "_" + str(NEURON_INDEX) + "_" + MODEL_PATH + ".png"
scipy.misc.imsave(out_filename, image)


