# Visualizing neurons one by one

1. Set layer and n_neurons in `vis_neuron.sh`
2. Run `vis_neuron.sh`

## Output: 
### Visualizations of single neurons 
Used frozen model files (downloaded automatically by the script)
  - googlenet_default.pb (network with Imagenet weights)
  - googlenet_finetuned.pb (Imagenet -> flowers transfer learning)
  - googlenet_flowers.pb (training on flowers)
  
  filename = {layer}\_{neuron id}\_{input frozen model file}.png
  
  All files are saved to working directory
  
### The three visualizations on one image
![alt text](https://github.com/robisz1911/LUCID_RESULTS/blob/master/neuron_catalog/Mixed4d_concat/Mixed4d_concat_merged/Mixed_4d_Concatenated-concat_179.png)


Find all the output visualizations [here](https://github.com/robisz1911/LUCID_RESULTS/tree/master/neuron_catalog)


  
  
