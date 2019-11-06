#!/bin/bash

# download necessary frozen model files

input=layers_name_channel_relu.txt
folder_for_visualizing=test_neuron_vis


mkdir $folder_for_visualizing
cp $input $folder_for_visualizing/$input
cp vis_neuron.py $folder_for_visualizing/vis_neuron.py
cp merge_neuron.py $folder_for_visualizing/merge_neuron.py
cd $folder_for_visualizing


curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_flowers.pb
curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_finetuned.pb
curl -OL https://raw.githubusercontent.com/robisz1911/LUCID_RESULTS/master/single_results/deepdream/googlenet_imagenet.pb
mv googlenet_imagenet.pb googlenet_default.pb


while IFS= read -r line
do

    vars=( $line )
    layer=${vars[0]}
#    n_neurons=${vars[1]}-1
    n_neurons=1

    dir=${layer////-}
    mkdir $dir


# for number of neurons in layer
    for (( i=0; i<=n_neurons; i++ ))
    do
        echo $layer/$i/$n_neurons
        for weights in default finetuned flowers
        do
            python vis_neuron.py --MODEL_PATH=googlenet_$weights.pb --LAYER=$layer --NEURON_INDEX=$i

        done

        python merge_neuron.py $layer $i
        rm -rf *default*.png
        rm -rf *finetuned*.png
        rm -rf *flowers*.png
        echo $layer"_"$i.png
        mv ${layer////-}"_"$i.png $dir/${layer////-}"_"$i.png

    done

#    mv $layer"_"$i.png $dir/$layer"_"$i.png

done < "$input"
