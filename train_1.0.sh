#!/bin/bash

folder_to_work_in=celeba_200epoch_15zoomin

batch_size=32
nb_epoch=200
dataset=celeba   # celeba / flowers17 / animals
cutoff=0
zoom_in=15




if [ $dataset == celeba ]
then
    trainer=train_celeba_1.0.py
else
    trainer=train_1.0.py
fi

mkdir $folder_to_work_in

python $trainer --batch_size=$batch_size --nb_epoch=$nb_epoch --dataset=$dataset --cutoff=$cutoff --zoom_in=$zoom_in

cp acc.txt $folder_to_work_in/
rm googlenetLucid.pb acc.txt model.ckpt model.ckpt.meta model.pb

for (( i = 0; i <= $nb_epoch; i++ ))
do

    cp $i.pb $folder_to_work_in/
    rm ${i}model.ckpt ${i}model.ckpt.meta $i.pb

done
