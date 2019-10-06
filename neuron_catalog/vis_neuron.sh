#!/bin/bash

# download necessary frozen model files
curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_flowers.pb
curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_finetuned.pb
curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_imagenet.pb
mv googlenet_imagenet.pb googlenet_default.pb


layer=Mixed_4d_Concatenated/concat

# for number of neurons in layer
for (( i=0; i<=511; i++ ))
do
    for weights in default finetuned flowers
    do

        python vis_neuron.py googlenet_$weights.pb $layer $i

    done

    python merge_neuron.py $layer $i

done


