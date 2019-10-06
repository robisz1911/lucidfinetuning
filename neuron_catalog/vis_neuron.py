import sys
import numpy as np
import scipy.misc
from lucid.modelzoo.vision_base import Model
import lucid.optvis.render as render
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param


# MODEL_PATH, LAYER, NEURON_INDEX = sys.argv[1:]
# 'inceptionLucid.pb', 'average_pooling2d_9/AvgPool'
MODEL_PATH, LAYER, NEURON_INDEX = sys.argv[1:]

NEURON_INDEX = int(NEURON_INDEX)

class FrozenNetwork(Model):
    model_path = MODEL_PATH
    image_shape = [299, 299, 3]
    image_value_range = (0, 1)
    input_name = 'input_1'


network = FrozenNetwork()
network.load_graphdef()

pixels = 256

param_f = lambda: param.image(pixels, fft=True, decorrelate=True)
obj = objectives.channel(LAYER, NEURON_INDEX)
images = render.render_vis(network, obj, param_f, thresholds=(2048,))
assert len(images)==1
image = images[0]
assert len(image)==1
image = image[0]

out_filename = LAYER.replace("/","-") + "_" + str(NEURON_INDEX) + "_" + MODEL_PATH + ".png"
scipy.misc.imsave(out_filename, image)

# for node in network_flowers.graph_def.node:
#     if layer in node.name and "Concat" in node.op:
#         print(node.input)

# if LAYER == "-":
#     images = []
#     layers = []
#     for l in sys.stdin:
#         layers.append(l.strip())
#
#     for layer in layers:
#         for i in range(COLUMNS):
#             obj = objectives.channel(layer, i)
#             renders = render.render_vis(network, obj)
#             assert len(renders)==1
#             image = renders[0]
#             assert len(image)==1
#             image = image[0]
#             images.append(image)
#     images = np.array(images)
#     height, width = 128, 128
#     rows = len(layers)
#     print(images.shape)
#     assert images.shape == (rows * COLUMNS, 128, 128, 3)
#     grid = (images.reshape(rows, COLUMNS, height, width, 3)
#               .swapaxes(1,2)
#               .reshape(height*rows, width*COLUMNS, 3))
#     scipy.misc.imsave(OUTPUT_PREFIX + ".png", grid)
#     sys.exit()
#
#
