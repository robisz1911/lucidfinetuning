LUCID visualization before and after finetuning.

1. What is LUCID visualization?
2. What is our research project about?
3. Datasets/Results.
4. What kind of tools available.


1. Highly recommend to read:
   https://distill.pub/2017/feature-visualization/
   which contains the basics of feature visualization.
   More details available from the original repository:
   https://github.com/tensorflow/lucid#recomended-reading
   
2. We study the visualization before and after finetuning.
   What happens in the visualization, when we finetune a neural network with imagenet weights, for a new image classification problem.
   
3. We save datasets/results in another repository since they are very big files.
   https://github.com/robisz1911/LUCID_RESULTS

4.
   4.1 Train/Visualization
           Trainging -> train.py
       Parameters in train.py:
           batch_size  : batch size
           nb_epoch    : number of epochs
           do_finetune : True -> training starts from imagenet weights
                         False -> training starts from random initialized weights
           cutoff      : number of the freezed layers ( for example cutoff = 5 -> the first 5 layers are not trainable )
           dataset     : changeable inside def load_data()  ( first line in the definition )
           topology    : the topology of the network
       Output:
           train.py generates a .pb file, which saves the network
           "topology".pb 
           
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
