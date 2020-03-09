#!/bin/bash

########## THIS SCRIPT WILL TRAIN THE NETWORK ( for [epoch_per_steps*steps] epochs  ) AND SAVE ITS' STATE AFTER EVERY $epoch_per_steps epochs  TO A .PB FILE ###########

# NOTE : SET THESE PARAMETERS BELOW!

foldername=folder_for_pbs # save .pb files into this folder
epoch_per_steps=1
batch_size=32
steps=10 # train [(steps-1)*epoch_per_steps] epochs //because 1.pb will belong to the state before training(original imagenet weights)
dataset=celeba   # celeba / flowers17 / animals
cutoff=0

if [ $dataset == celeba ]
then
    trainer=train_celeba_final.py
else
    trainer=train.py
fi

mkdir $foldername

for (( i = 1; i <= $steps; i++ ))
do 
    if [ $i == 1 ]; then
        python $trainer --batch_size=$batch_size --nb_epoch=$epoch_per_steps --do_load_model=False --do_save_model=True --do_finetune=False --dataset=$dataset --cutoff=$cutoff
        cp googlenetLucid.pb $i.pb
        mv $i.pb $foldername
    fi
    if [ $i  == 2 ]; then
        python $trainer --batch_size=$batch_size --nb_epoch=$epoch_per_steps --do_load_model=False --do_save_model=True --do_finetune=True --dataset=$dataset --cutoff=$cutoff
        cp googlenetLucid.pb $i.pb
        mv $i.pb $foldername
    fi
    if [ $i != 2 ] && [ $i != 1 ]; then
        python $trainer --batch_size=$batch_size --nb_epoch=$epoch_per_steps --do_load_model=True --do_save_model=True --do_finetune=True --dataset=$dataset --cutoff=$cutoff
        cp googlenetLucid.pb $i.pb
        mv $i.pb $foldername
    fi
    echo "------------------------------------------------------------------------------------"
    echo $i"-th epoch"
    echo "------------------------------------------------------------------------------------"
done
