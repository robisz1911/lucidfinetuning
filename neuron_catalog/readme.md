# Visualizing neurons one by one

1. Set input in `vis_neuron.sh` and `merge_neuron.sh`: a text file containing the name - number of channels pairs for the layers to be visualized
2. Run `vis_neuron.sh` creates single neuron visaulizations for all neurons of the layers in input file
3. Run `merge_neuron.sh` for merged images seen below - merges all visualizations of the same neuron from different frozen model files

## Output: 
### Visualizations of single neurons 
Used frozen model files (downloaded automatically by the script)
  - googlenet_default.pb (network with Imagenet weights)
  - googlenet_finetuned.pb (after Imagenet -> flowers transfer learning)
  (- googlenet_flowers.pb (training on flowers) )
  
  Use of arbitrary frozen model file (e.g animals):
  - Copy googlenet_animals.pb file to working directory
  - Add animals to line 33: 
    ```
    for weights in default finetuned animals
    ```
  
Saved to /layer/layer\_neuron id\_frozen model file.png
  
### All visualizations of a neuron on one image
![alt text](https://github.com/robisz1911/LUCID_RESULTS/blob/master/neuron_catalog/Mixed4d_concat/Mixed4d_concat_merged/Mixed_4d_Concatenated-concat_179.png)

Saved to /layer/layer_merged 


Find all the output visualizations [here](https://github.com/robisz1911/LUCID_RESULTS/tree/master/neuron_catalog)


  
  
