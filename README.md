## LUCID visualization before and after finetuning.

1. What is LUCID visualization?
2. What is our research project about?
3. Datasets/Results.
4. What kind of processes available.

##### Detailed:
1. Highly recommend to read:
 - https://distill.pub/2017/feature-visualization/
   which contains the basics of feature visualization
 - https://github.com/tensorflow/lucid#recomended-reading
   More details at the original repository
2. We study the visualization before and after finetuning.
 - What happens in the visualization, when we finetune a neural network with imagenet weights, for a new image classification problem.
   
3. We save datasets/results in another repository since they are very big files.
 - https://github.com/robisz1911/LUCID_RESULTS

4. Processes<br/>
      Train/Visualization<br/>
       - Trainging -> train.py<br/>
      Parameters in train.py:<br/>
        - batch_size  : batch size<br/>
        - nb_epoch    : number of epochs<br/>
        - do_finetune : True -> training starts from imagenet weights | False -> training starts from random initialized weights<br/>
        - cutoff      : number of the freezed layers ( for example cutoff = 5 -> the first 5 layers are not trainable )<br/>
        - dataset     : changeable inside def load_data()  ( first line in the definition )<br/>
        - topology    : the topology of the network<br/>
      Output:<br/>
        - train.py generates a .pb file, which saves the network<br/>
        - "topology".pb <br/>
           
      Train/Visualization by steps<br/>
      

# WHAT ARE THESE SCRIPTS FOR? (vis_script.sh | train_script.sh | train_vis_script.sh)
# to train and visualize by steps | to visualize the network every 5 epochs for example, we save the network's condition after each 5 epochs, save these conditions into .pb files, and visualize them, finally we want to see all the pictures on the same page, so with merge.py, we create one big image


# HOW TO USE BASH SCRIPTS(.sh)
# training and visualizing --> train_vis_script.sh
# only training            --> train_script.sh
# only visualizing         --> vis_script.sh
# give execute permission to your script:
#      chmod +x /path/to/yourscript.sh   or chmod +x ./yourscript.sh (if u're in the same folder)
# run script:
#      /path/to/yourscript.sh            or ./yourscript.sh (if u're in the same folder)


# DETAILED DESCRIPTION IN COMMENTS ( read train_vis_script.sh's comments first )
 
# NOTE : error:  /bin/bash^M: bad interpreter: No such file or directory    (notepad++ saved the files with dos endings instead of unix)
#        solution:  vim valami,sh  ->and run    :set fileformat=unix   save it again(:wq)




# 3. How to train and/or visualize with the existing tools.
First of all clone our github repository. (https://github.com/robisz1911/lucidfinetuning)

## 3.1. Train
There are different ways to train with the existing tools.
##### 3.1.1. Train and save only the final state of the network
The first one is to train the network, and save only its’ final state into a .pb file, the .pb file is the one, we can use to generate visualization from.
We have another github repository ( https://github.com/robisz1911/datasets ) from where you can download flowers or animals dataset, but our main focus is on flowers17 right now.
Download it to your renyi computer, into the same folder as lucid files.
Run vim train.py to edit the parameters of the training.
Parameters what are interesting for us:
•	nb_epochs ->number of epochs<br/>
•	batch_size	->batch size<br/>
•	cutoff	the numbers of the first x layers what are not trainable<br/>
can be found in def finetune at line109(L109)<br/>
•	compile parameters	->optimizer, learning rate(L117)<br/>
•	dataset	default setup is flowers17<br/>
can be found in the first line of def load_data(L126)<br/>
•	topology	->the network’s topology<br/>
options are googlenet, inceptionv3 and vgg16<br/>
•	do_finetune	->if True training starts from imagenet weights<br/>
	if False no training at all<br/>
Run train.py with the command python train.py which will train the network and save its’ final state into a .pb file.

##### 3.1.2. Train and save the network’s states after each x epochs
This tool can be found in train_script.sh. Bash script format is .sh, and the first line should be #!/bin/bash as you can see by running vim train_script.sh.
You have to change the folder name, because a new folder should be generated where the .pb files will be saved. (mkdir training_x)
Another parameter in the script is rows, which means the overall epochs equals to (rows-1)*nb_epoch. Rows are the number of iterations of the training. Each iteration we train the network for nb_epoch.(except for the first one, where we save the network before training) For more details check the comments in the script.
Before run a bash script, you have to give execute permission to it with the command chmod +x /path/to/yourscript.sh or chmod +x ./yourscript.sh (if you are at its’ folder).
Run script with the command /path/to/yourscript.sh or ./yourscript.sh.
#####holnap csekkolni, hogy a githubos scriptek unix formatban vannak-e mentve########
This will generate the .pb files into the created folder.
(1.pb belongs to the network before training, x.pb is the state after [(x-1)*nb_epoch] epochs)

## 3.2. Visualize
The visualizations created by vis.py which input is a .pb file.
In vis.py we can change COLUMNS what is the number of neurons being visualized.
##### 3.2.1. Visualize only one .pb file(one state of the network).
Run the command:
cat googlenet-node-names | grep „expression” | python vis.py sample.pb – grid sample_visualization
Googlenet-node-names contains the layer names of the googlenet architecture, and grep command select all of these layers which include „expression”.
Then the visualization is generated for the selected layers. (the first COLUMNS neurons are visualized in each layers)
The output is the sample_visualization.png.
##### 3.2.2. Visualize from many .pb files.
If there was a training where the network’s states where saved in y iterations, then we have a given folder(sample_folder) with y .pb files, from 1.pb to y.pb.
Set foldername to a new name, where the images will be saved.
Set rows equals to the number of .pb files. Name the layer you want to visualize by setting layer.
We can run vis_script.sh with the command ./vis_script.sh to visualize a selected layer through every state of its training.
It will run vis.py on every .pb files in the folder, and it generates y .png files, what are the 
visualizations of the selected layer in each state.
The script will then run merge.py, what concatenate these pictures into one big image, which first row belongs to the 1. state of the network(before training state), and the last row belongs to the y. state. Read merge.py for more details.
## 3.3. Train and visualize
This script is the combination of the ones above. (train_script.sh and vis_script.sh)
It’ll do the training and visualizing in each iteration (except for the first one, where only the visualization), then concatenate the pictures with merge.py.
Set folder_for_pictures, folder_for_trainings, rows and layer, just as mentioned above.
Run train_vis_script.sh with the command ./train_vis_script.sh.

