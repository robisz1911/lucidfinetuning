#!/bin/bash

# download necessary frozen model files
# curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_flowers.pb
curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_finetuned.pb
curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_imagenet.pb
mv googlenet_imagenet.pb googlenet_default.pb

input="layers_name_channel_concat.txt"

while IFS= read -r line
do

    vars=( $line )
    layer=${vars[0]}
    n_neurons=${vars[1]}-1

    echo $layer
    echo $n_neurons

    dir=${layer////-}

    if [ ! -d "$dir" ]
    then
        mkdir $dir
    fi


# for number of neurons in layer
    for (( i=0; i<=n_neurons; i++ ))
    do
        echo $i
        for weights in default finetuned #flowers
        do
            python vis_neuron.py --MODEL_PATH=googlenet_$weights.pb --LAYER=$layer --NEURON_INDEX=$i

        done

        #python merge_neuron.py $layer $i

    done

    mv ${dir}_* $dir

done < "$input"
