import tensorflow.compat.v1 as tf
import numpy as np
from numpy import savez_compressed
from numpy import savetxt
import skimage.io
import sys


# Utilities

def avg_pool(input):
  return np.mean(input, axis=(1,2))

def max_pool(input):
  #print(input.shape)
  maxed2 = np.max(input, axis=(1,2))
  #maxed2 = np.max(maxed1, axis=1)
  #print(maxed2.shape)
  return maxed2

def list_tensors(graph):
  i = 0
  tensor_list = []
  for op in graph.get_operations():
    if 'convolution' in op.name and not 'convolution_' in op.name:
      #print(op.name)
      tensor_list.append(op.name + ':0')
    i = i + 1
  return tensor_list

def list_tensor_start_indices(graph):
  i = 0
  indices = []
  for op in graph.get_operations():
    if 'convolution' in op.name and not 'convolution_' in op.name:
      tensor = graph.get_tensor_by_name(op.name + ':0')
      indices.append(i)
      i = i + tensor.shape[3];
  return indices

def index_to_neuron(graph, index):
  i = 0
  for op in graph.get_operations():
    if 'convolution' in op.name and not 'convolution_' in op.name:
      tensor = graph.get_tensor_by_name(op.name + ':0')
      i = i + tensor.shape[3];
      if index < i:
        print(op.name + ' --- Neuron #' + str( index - (i-tensor.shape[3]) ))
        return


def image_list_to_npy(filenames, output):
    imgs = []
    for filename in filenames:
        img = skimage.io.imread(filename)
        imgs.append(img)
    imgs = np.array(imgs)
    np.save(output, imgs)
    return imgs


def main_convert():
    # corpus = "celeba" ; kind = "finetuned"
    corpus, kind, output = sys.argv[1:]

    filenames = []
    for l in sys.stdin:
        batch, layer, size = l.strip().split()
        size = int(size)
        for indx in range(size):
            filename = "%s_%s/%s/%s_%d_googlenet_%s.pb.png" % (corpus, batch, layer, layer, indx, kind)
            filenames.append(filename)
    print("\n".join(filenames))
    image_list_to_npy(filenames, output)


def main_convert_old():
    raise Exception("not safe, use main_convert() instead")
    filenames = [l.strip() for l in sys.stdin]
    parseds = [filename.split("/")[-1].split("_") for filename in filenames]
    parseds = [(parsed[:7], int(parsed[7]), filename) for (filename, parsed) in zip(filenames, parseds)]
    parseds = sorted(parseds, key=lambda k: (k[0], k[1]))
    filenames_sorted = [filename for (_, _, filename) in parseds]
    print("\n".join(filenames_sorted))
    output = sys.argv[1]
    image_list_to_npy(filenames_sorted, output)


# main_convert() ; sys.exit()


image_file = 'lucid-celeba-full-256.npy'
# image_file = 'lucid-default-imagenet-full-256.npy'
print("loading", image_file)
image_list = np.load(image_file)

# image_list = image_list[-128:] ; print("truncating image set to last 128")
image_list = image_list[:1280] ; print("truncating image set to first 1280")


image_list = image_list[:, :224, :224, :] ; print("truncating image sizes to 224x224")


# which_network = "celeba"
which_network = "imagenet"
print("using", which_network, "weigths")
if which_network == "imagenet":
    pb_file = "1.pb"
elif which_network == "celeba":
    pb_file = "200.pb"
else:
    assert False


# Reading network
sess = tf.Session()
with open(pb_file, "rb") as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  sess = tf.Session(tf.import_graph_def(graph_def, name=''))

# import graph_def
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

# Getting the list of convolutional tensors
tensors = list_tensors(sess.graph)

image_counter = 0
mx = np.random.rand(0,0)
batch_size = 128
batch_count = 10 # 57
print('Computing results for ' + str(batch_size * batch_count) + ' images')
print('Batch size: ' + str(batch_size))
for image_counter in range(batch_count):
  # Creating a batch of images
  print('Working on images: ' + str(image_counter*batch_size+1) + '-' + str((image_counter+1)*batch_size))
  image_batch = image_list[image_counter*batch_size:(image_counter+1)*batch_size]
  print(image_batch.shape)
  
  # Prediction
  preds = sess.run(tensors, {"input_1:0": image_batch})

  # Collecting all neurons into a block (nr_images x batch size)
  image_line = np.random.rand(batch_size, 0)
  #print(len(preds))
  for pred in preds: # For each layer
    maxed = max_pool(pred)
    #print(maxed)
    image_line = np.hstack((image_line, maxed))

  # Stacking blocks
  if mx.shape == (0,0):
    mx = image_line
  else:
    mx = np.vstack((mx,image_line))

'''
print("Saving matrix...")
savez_compressed('mx-ours.npz', mx)
print(mx.shape)
'''

mx_top = mx.T[:1280, :]
print(mx_top.shape)

mx_top -= mx_top.mean(axis=1, keepdims=True)

print(np.mean(mx_top.diagonal()))
for i in range(10):
  s = mx_top.copy()
  np.random.shuffle(s)
  print(np.mean(s.diagonal()))

print(np.mean(s))



vis = True
if vis:
    import matplotlib.pyplot as plt
    plt.imshow(mx_top, cmap='hot', interpolation='nearest')
    plt.show()
