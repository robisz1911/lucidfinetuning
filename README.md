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
