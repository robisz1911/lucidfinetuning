import tensorflow as tf

# extract and write  node names of saved graph (= graph_file) to txt 

graph_file = "inceptionv3Lucid.pb"
graph_def = tf.GraphDef()
with open(graph_file, "rb") as f:
  graph_def.ParseFromString(f.read())
txt_f = open(graph_file+"_node_names.txt", "w+")
for node in graph_def.node:
  txt_f.write(str(node.name)+ "\r\n")
txt_f.close()
