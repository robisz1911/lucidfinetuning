# 3. How to train and/or visualize with the existing tools.
First of all clone our github repository. (https://github.com/robisz1911/lucidfinetuning)<br/>
• Train and save only the final state of the network<br/>
• Train and save the network’s states after each x epochs<br/>
• Visualize only one .pb file(one state of the network)<br/>
• Visualize from many .pb files<br/>
• Train and visualize
## 3.1. Train
There are different ways to train with the existing tools:<br/>

##### 3.1.1. Train and save only the final state of the network
The first one is to train the network, and save only its’ final state into a .pb file, the .pb file is the one, we can use to generate visualization from.
We have another github repository ( https://github.com/robisz1911/datasets ) from where you can download flowers or animals dataset, but our main focus is on flowers17 right now.
Download it to your renyi computer, into the same folder as lucid files.
Run vim train.py to edit the parameters of the training.
Parameters what are interesting for us:
•	nb_epochs ->number of epochs<br/>
•	batch_size	->batch size<br/>
•	cutoff	->the numbers of the first x layers what are not trainable<br/>
can be found in def finetune at line109(L109)<br/>
•	compile parameters	->optimizer, learning rate(L117)<br/>
•	dataset	->default setup is flowers17<br/>
can be found in the first line of def load_data(L126)<br/>
•	topology	->the network’s topology<br/>
options are googlenet, inceptionv3 and vgg16<br/>
•	do_finetune	->if True training starts from imagenet weights / if False no training at all<br/>
Run train.py with the command python train.py which will train the network and save its’ final state into a .pb file.

##### 3.1.2. Train and save the network’s states after each x epochs
This tool can be found in train_script.sh. Bash script format is .sh, and the first line should be #!/bin/bash as you can see by running vim train_script.sh.<br/>
You have to change the foldername, because a new folder should be generated where the .pb files will be saved. (name it to a non-existing foldername)<br/>
Another parameter in the script is rows, which means the overall epochs equals to (rows-1)*nb_epoch. Rows are the number of iterations of the training. Each iteration we train the network for nb_epoch.(except for the first one, where we save the network before training)<br/>
For more details check the comments in the script.<br/>
Before run a bash script, you have to give execute permission to it with the command chmod +x /path/to/yourscript.sh or chmod +x ./yourscript.sh (if you are at its’ folder).<br/>
Run script with the command /path/to/yourscript.sh or ./yourscript.sh.<br/>
This will generate the .pb files into the created folder.<br/>
(1.pb belongs to the network before training, x.pb is the state after [(x-1)*nb_epoch] epochs)<br/>

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
This script(train:vis:script.sh) is the combination of the ones above. (train_script.sh and vis_script.sh)
It’ll do the training and visualizing in each iteration (except for the first one, where only the visualization), then concatenate the pictures with merge.py.
Set folder_for_pictures, folder_for_trainings, rows and layer, just as mentioned above.
Run train_vis_script.sh with the command ./train_vis_script.sh.

##### NOTE: #####
if you face error:  /bin/bash^M: bad interpreter: No such file or directory<br/>
(notepad++ saved the files with dos endings instead of unix)<br/>
solution:  vim yourscript.sh -> run: set fileformat=unix -> save it with :wq
