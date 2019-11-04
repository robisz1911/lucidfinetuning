# Creating visualizations before and after transfer learning using the Lucid library

The [lucid library](https://github.com/tensorflow/lucid) and the [Feature Visualization article](https://distill.pub/2017/feature-visualization/)

## Single visualizations

We can visualize neurons of the network [before](https://github.com/robisz1911/LUCID_RESULTS/blob/master/single_results/googlenet_default.png) and [after](https://github.com/robisz1911/LUCID_RESULTS/blob/master/single_results/flowers/flowers_20cutoff.png) transfer learning.

### Training
The script `train.py` saves the network's final state to a .pb file. It is a frozen model file created by the script, that contains all the relevant information (model.pb and checkpoint file) of the network. The Lucid library requires frozen model files of this kind in order to create visualizations.
We have [another repository](https://github.com/robisz1911/LUCID_RESULTS) from where you can download the Flowers17 and animals datasets and take a look at some of the results.

#### Important parameters of `train.py`:
 - *nb_epochs* - number of epochs (line 32)
 - *batch_size* - batch size (line 31)
 - *cutoff* - the numbers of the first x layers what are not trainable (line 109)
 - *compile parameters - optimizer, learning rate* (line 117)
 - *dataset* - default setup is flowers17. (line 126)
 - *topology* - the network’s topology (line 161)
                options are googlenet, inceptionv3 and vgg16
                setting the weights: „imagenet” or „None” for random initialization
 - *do_finetune* - whether to perform training/transfer learning (line 178)
                   if True: performs transfer learning
                   if False: no transfer learning, we save the network with the original weights (ImageNet or random). 
                   For example if you want a frozen model file with the ImageNet weights, set do_finetune=False and run                   
                   the file – it is useful for comparison

Run `train.py` to perform transfer learning, and save the network's final state into a .pb file.

### Visualization
The visualizations are created by `vis.py`, which needs the name(s) of the nodes we want to visualize, a .pb file, and a string that will be the name of the saved image as inputs. Here the main parameter is *COLUMNS* is the number of neurons visualized from each layer.
Also, this is where we can set the parameters, and objectives for the Lucid visualization.

In order to visualize one certain state of the network, run the commands:
```
cat googlenet-node-names | grep „expression” | python vis.py sample.pb – sample_visualization COLUMNS
```
`Googlenet-node-names` contains the layer names of the googlenet architecture, and grep
command select all of theose layers which include *„expression”*. Then the visualization of the
first *COLUMNS* neurons of the chosen layers is generated. The output image (*sample_visualization.png*) 
is saved to the working directory. Its rows correspond to the layers, and the columns correspond to neurons.

## Step by step training and visualization

We can also [track the changes](https://github.com/robisz1911/LUCID_RESULTS/blob/master/steps_results/PICTURES/flowers_20cutoff_50epochs_mixed4c_png/merged.png) throughout the transfer learning process. 
Here we can see the first five neurons of the *Mixed_4c_Branch_3_b_1x1_act/Relu* layer (columns) of GoogLeNet during the 
first 50 epochs (rows). 

### Training by steps
This tool can be found in `train_script.sh`. Here the name of the directory - where all the .pb
files of the current run will be saved - is a variable, which you can change at the beginning of
the script (currently *training_x*)
Another parameter in the script is *rows*, referring to the number of rows the final image will
have. It also means that the overall number of epochs we run the training for equals to *(rows-
1)nb_epoch*. In other words, rows is the number of iterations of the training. At each iteration
we train the network for *nb_epoch* epochs.(except for the first one, where we save the
network before training – with its original weights) For more details check the comments in
the script.


### Visualization from several .pb files.
If there was a previous training where the network states were saved through y iterations, then
we have a given folder (*sample_folder* – it is what you set in the beginning of `train_script.sh`)
containing the .pb files, from 1.pb to y.pb.
Set *foldername* (in `vis_script.sh`) to create a new directory where the images will be saved. Set
*rows* equal to the number of .pb files to use all of them in the visualization. Note, that it is not
necessary to use all the existing .pb files, you can just use the first let’s say 10 of them, by
setting rows=10 here. It is only necessary that the rows parameter is less or equal to the
number of .pb files in the directory. Name the layer you want to visualize by setting layer.

The script works the following way: it will run `vis.py` on every .pb file in the folder, and thus
it generates the y.png files, which are the visualizations of the selected layer at the yth step.
The script will then run `merge.py`, which concatenates these pictures into one big image,
where the first row belongs to the default state of the network (before training).


### Training and visualization at once
The script `train_vis_script.sh` is the combination of the ones above. (`train_script.sh` and
`vis_script.sh`)
It does the training and visualization at each iteration (except for the first one, where we wish
to see the network’s pre-training state, meaning the script only performs visualization but not
training), then concatenates the pictures with `merge.py`.
 - 1.pb is the default network – imagenet (or random initialized) weights
 - 1.png is the chosen layer visualized with these weights
 - 2.pb is the network’s state after *nb_epoch* epochs
 - 2.png is the choosen layer visualized after *nb_epoch* epochs
 - n.pb is the network’s state after *(n-1)nb_epoch* epochs
 - n.png is the choosen layer visualized after *(n-1)nb_epoch* epochs and so on

Set *folder_for_pictures*, *folder_for_trainings*, *rows* and *layer*, just as mentioned above and run `train_vis_script.sh`.
