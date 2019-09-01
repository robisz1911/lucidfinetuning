#!/bin/bash

# THIS SCRIPT DOES:
#     make training_x and pictures_x folders
#     copy merge.py into pictures_x      | read merge.py's comments for description
#     do the training - .pb files will be saved into training_x
#     visualize chosen layer and create one image - .png files, and merged.png saved in pictures_x
# 
# in merge.png:      1.pb is the default - imagenet                   | 1.png is the choosen layer visualized from imagenet
#                    2.pb the network after (nb_epochs) epochs        | 2.png is the choosen layer visualized after (nb_epochs) epochs
#                    n.pb the network after (n-1)*(nb_epochs) epochs  | n.png is the choosen layer visualized after (nb_epochs)*(n-1) epochs
#
# commands      using sed -i (search and reaplace) you can set all the parameters from this script - or set them manually in train.py, vis.py, merge.py
#               .sh files should have execute permission    |   chmod +x ./yourscriptname.sh
#               run .sh file                                |            ./yourscriptname.sh  


######  NOTE : before running the script, delete training_x, pictures_x folders ###################################################


mkdir training_x
mkdir pictures_x
cp merge.py pictures_x/merge.py

rows=50
layer=Mixed_4f_Branch_3_b_1x1_act/Relu


#sed -i 's/nb_epoch = .*/nb_epoch = 1/' train.py   # SET : visualize after every n'th epochs
#sed -i 's/COLUMNS = .*/COLUMNS = 5/' vis.py       # SET : number of neurons visualized       |THESE TWO-      |
#sed -i 's/columns = .*/columns = 5/' merge.py     # SET : number of neurons visualized       |MUST BE THE SAME|

for (( i = 1; i <= $rows; i++ ))                                    # SET : train for (np_epochs) epochs then visualize this many times
do
    
    if [ $i == 1 ]; then                            # both variables set to False, so we do not finetune or load weights from previous training 
        sed -i 's/do_load_model = .*/do_load_model = False/' train.py
        sed -i 's/do_finetune = .*/do_finetune = False/' train.py      
    fi
    if [ $i  == 2 ]; then                           # finetune = True, we start finetuning the model, but still not loading weights from previous training
        sed -i 's/do_finetune = .*/do_finetune = True/' train.py
        sed -i 's/do_load_model = .*/do_load_model = False/' train.py
    fi
    if [ $i != 2 ] && [ $i != 1 ]; then             # finetune for nb_epochs, and load the weights from the previous iteration
        sed -i 's/do_finetune = .*/do_finetune = True/' train.py        
        sed -i 's/do_load_model = .*/do_load_model = True/' train.py
    fi
                                                    
    python train.py
    cat googlenet-node-names | grep $layer | python vis.py googlenetLucid.pb - $i
    cp googlenetLucid.pb $i.pb
    mv $i.pb training_x
    mv $i.png pictures_x

done
                                                    # finally, we cd into pictures_x folder and run merge.py ( this should be executed next to the images )

cd pictures_x
python merge.py


