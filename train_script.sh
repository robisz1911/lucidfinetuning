#!/bin/bash

########## THIS SCRIPT WILL TRAIN THE NETWORK AND SAVE ITS' STATE IN EVERY ITERATION TO A .PB FILE ###########

###############  NOTE : before running rename foldername to a non existing folder#############################

#SET NUMBER OF EPOCHS(nb_epoch) and BATCH SIZE(batch_size) in train.py by setting the two variables below(uncomment them, and the two sed lines)
#set_nb_epoch_in_train = 1
#set_batch_size_in_train = 32

# with the command sed, it's possible to change parameters in different files from anywhere
#sed -i 's/nb_epoch = .*/nb_epoch = $set_nb_epoch_in_train/' train.py
#sed -i 's/batch_size = .*/batch_size = $set_batch_size_in_train/' train.py

foldername=training_x
columns=5
nb_epoch=1
batch_size=32
mkdir $foldername

rows=100 # train [(rows-1)*nb_epoch] epochs because 1.pb will belong to the state before training(when finetuning, 1.pb have the original imagenet weights)

for (( i = 1; i <= $rows; i++ ))
do
    
    if [ $i == 1 ]; then                            # both variables set to False, so we do not finetune or load weights from previous training
        #sed -i 's/do_load_model = .*/do_load_model = False/' train.py
        #sed -i 's/do_finetune = .*/do_finetune = False/' train.py
        python train.py $batch_size $nb_epoch 0 1 0
        cp googlenetLucid.pb $i.pb
        mv $i.pb $foldername
    fi
    if [ $i  == 2 ]; then                           # finetune = True, we start finetuning the model,but still not loading weights from previous training
        #sed -i 's/do_finetune = .*/do_finetune = True/' train.py
        #sed -i 's/do_load_model = .*/do_load_model = False/' train.py
        python train.py $batch_size $nb_epoch 0 1 1
        cp googlenetLucid.pb $i.pb
        mv $i.pb $foldername

    fi
    if [ $i != 2 ] && [ $i != 1 ]; then             # finetune for nb_epochs, and load the weights from the previous iteration
        #sed -i 's/do_finetune = .*/do_finetune = True/' train.py        
        #sed -i 's/do_load_model = .*/do_load_model = True/' train.py
        python train.py $batch_size $nb_epoch 1 1 1
        cp googlenetLucid.pb $i.pb
        mv $i.pb $foldername

    fi
                                                    # each iteration we train, then save and copy frozen model file to training_x

done
