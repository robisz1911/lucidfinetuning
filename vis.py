import sys
import numpy as np
import scipy.misc
from lucid.modelzoo.vision_base import Model
import lucid.optvis.render as render
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param


MODEL_PATH, LAYER, OUTPUT_PREFIX, COLUMNS_arg = sys.argv[1:]
# 'inceptionLucid.pb', 'average_pooling2d_9/AvgPool'

COLUMNS = int(COLUMNS_arg)
print(COLUMNS)
#COLUMNS = 5

class FrozenNetwork(Model):
    model_path = MODEL_PATH
    image_shape = [299, 299, 3]
    image_value_range = (0, 1)
    input_name = 'input_1'

network = FrozenNetwork()
network.load_graphdef()

if LAYER == "-":
    height, width = 144, 144
    images = []
    layers = []
    for l in sys.stdin:
        layers.append(l.strip())

    for layer in layers: 
        for i in range(COLUMNS):
            param_f = lambda: param.image(height, fft=True, decorrelate=True)
            obj = objectives.channel(layer, i)
            renders = render.render_vis(network, obj, param_f, thresholds=(2048,))
            assert len(renders)==1
            image = renders[0]
            assert len(image)==1
            image = image[0]
            images.append(image)
    images = np.array(images)
    rows = len(layers)
    print(images.shape)
    assert images.shape == (rows * COLUMNS, height, width, 3)
    grid = (images.reshape(rows, COLUMNS, height, width, 3)
              .swapaxes(1,2)
              .reshape(height*rows, width*COLUMNS, 3))
    scipy.misc.imsave(OUTPUT_PREFIX + ".png", grid)
    sys.exit()


for i in range(COLUMNS):
    obj = objectives.channel(LAYER, i)
    images = render.render_vis(network, obj)
    assert len(images)==1
    image = images[0]
    assert len(image)==1
    image = image[0]
    scipy.misc.imsave("%s_%0d.png" % (OUTPUT_PREFIX, i), image)
