python /home/csadrian/venvs/p36/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
  --input_graph=model.pb \
  --input_checkpoint=./model.ckpt \
  --input_binary=true \
  --output_graph=googlenetLucid.pb \
  --output_node_name=Mixed_5c_Concatenated/concat
