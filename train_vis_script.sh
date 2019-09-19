#!/bin/bash

# THIS SCRIPT DOES:
#     create one folder for images, and one for training files (folder_for_pictures,folder_for_trainings)
#     copy merge.py into folder_for_pictures      |    read merge.py's comments for more details
#     do the training - .pb files will be saved into training_x
#     visualize chosen layer and create one image  - .png files, and merged.png saved in pictures_x
# 
#         1.pb is the default - imagenet                     | 1.png is the choosen layer visualized from imagenet
#         2.pb the network after (nb_epochs) epochs          | 2.png is the choosen layer visualized after (nb_epochs) epochs
#         n.pb the network after [(n-1)*(nb_epochs)] epochs  | n.png is the choosen layer visualized after [(nb_epochs)*(n-1)] epochs
#
# commands      using sed -i (search and reaplace) you can set all the parameters from this script
#               .sh files should have execute permission    |   chmod +x ./yourscriptname.sh
#               run .sh file                                |            ./yourscriptname.sh  

######  NOTE : before running the script, rename folder_for_pictures and folder_for_trainings to non existing foldernames ############

#sed -i 's/nb_epoch = .*/nb_epoch = 1/' train.py   # SET : visualize after every n'th epochs
#sed -i 's/COLUMNS = .*/COLUMNS = 5/' vis.py       # SET : number of neurons visualized       |THESE TWO-      |
#sed -i 's/columns = .*/columns = 5/' merge.py     # SET : number of neurons visualized       |MUST BE THE SAME|

folder_for_pictures=pictures_x
folder_for_trainings=training_x
rows=50

mkdir $folder_for_pictures
mkdir $folder_for_trainings
cp merge.py $folder_for_pictures/merge.py

layer=Mixed_4f_Branch_3_b_1x1_act/Relu



for (( i = 1; i <= $rows; i++ ))
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
    mv $i.pb $folder_for_trainings
    mv $i.png $folder_for_pictures

done

cd $folder_for_pictures
python merge.py



