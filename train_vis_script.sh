#!/bin/bash

# TRAIN AND VISUALIZE

# NOTE : SET THE PARAMETERS BELOW :

folder_for_pictures=xxxxxpics
folder_for_trainings=xxxxpb
steps=2 # # train [(steps-1)*epoch_per_steps] epochs //because 1.pb will belong to the state before training(original imagenet weights) and visualize after each steps
batch_size=32
epoch_per_steps=1
columns=2                                      # the first $columns neurons 'll be visualized
layer=Mixed_4f_Branch_3_b_1x1_act/Relu         # layer names can be found in googlenet-node-names
dataset=flowers17
cutoff=0

mkdir $folder_for_pictures
mkdir $folder_for_trainings
cp merge.py $folder_for_pictures/merge.py

for (( i = 1; i <= $steps; i++ ))
do
    
    if [ $i == 1 ]; then
        python train.py --batch_size=$batch_size --nb_epoch=$epoch_per_steps --do_load_model=False --do_save_model=True --do_finetune=False --dataset=$dataset --cutoff=$cutoff
        cat googlenet-node-names | grep $layer | python vis.py googlenetLucid.pb - $i $columns
        cp googlenetLucid.pb $i.pb
        mv $i.pb $folder_for_trainings
        mv $i.png $folder_for_pictures
    fi
    if [ $i  == 2 ]; then
        python train.py --batch_size=$batch_size --nb_epoch=$epoch_per_steps --do_load_model=False --do_save_model=True --do_finetune=True --dataset=$dataset --cutoff=$cutoff
        cat googlenet-node-names | grep $layer | python vis.py googlenetLucid.pb - $i $columns
        cp googlenetLucid.pb $i.pb
        mv $i.pb $folder_for_trainings
        mv $i.png $folder_for_pictures

    fi
    if [ $i != 2 ] && [ $i != 1 ]; then
        python train.py --batch_size=$batch_size --nb_epoch=$epoch_per_steps --do_load_model=True --do_save_model=True --do_finetune=True --dataset=$dataset --cutoff=$cutoff
        cat googlenet-node-names | grep $layer | python vis.py googlenetLucid.pb - $i $columns
        cp googlenetLucid.pb $i.pb
        mv $i.pb $folder_for_trainings
        mv $i.png $folder_for_pictures

    fi

done

cd $folder_for_pictures
python merge.py --column=$columns



