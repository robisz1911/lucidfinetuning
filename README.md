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
   4.1 Train/Visualization<br/>
       - Trainging -> train.py
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
           
   4.2 Train/Visualization by steps

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
