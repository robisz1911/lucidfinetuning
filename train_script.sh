#!/bin/bash

######  NOTE : before running the script, delete training_x folder or rename it to a non existing folder#############################

#sed -i 's/nb_epoch = .*/nb_epoch = 1/' train.py    # SET : visualize after every n'th epochs
foldername = training_x
mkdir $foldername

rows=100 # train 100*nb_epoch

for (( i = 1; i <= $rows; i++ ))
do
    
    if [ $i == 1 ]; then                            # both variables set to False, so we do not finetune or load weights from previous training
        sed -i 's/do_load_model = .*/do_load_model = False/' train.py
        sed -i 's/do_finetune = .*/do_finetune = False/' train.py
    fi
    if [ $i  == 2 ]; then                           # finetune = True, we start finetuning the model,but still not loading weights from previous training
        sed -i 's/do_finetune = .*/do_finetune = True/' train.py
        sed -i 's/do_load_model = .*/do_load_model = False/' train.py
    fi
    if [ $i != 2 ] && [ $i != 1 ]; then             # finetune for nb_epochs, and load the weights from the previous iteration
        sed -i 's/do_finetune = .*/do_finetune = True/' train.py        
        sed -i 's/do_load_model = .*/do_load_model = True/' train.py
    fi
                                                    # each iteration we train, then save and copy frozen model file to training_x
    python train.py
    cp googlenetLucid.pb $i.pb
    mv $i.pb $foldername

done
