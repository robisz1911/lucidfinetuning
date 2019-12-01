#!/bin/bash

input="layers_neuron_list.txt"
number_of_pb_files=20
savefolder=output_merged_pictures_threshold_4096

mkdir $savefolder

while IFS= read -r line
do
#read txt line by line
#each line is a space separated string
#example line:    "layer 1,2,3,4,5"
#separate by space into 2 variable
    vars=( $line )
    layer=${vars[0]}
    neuron_list=${vars[1]}
#iterate over the second variable below(neuron indexes)
    for neuron_index in $(echo $neuron_list | sed "s/,/ /g")
    do
        #echo "$layer"
        #echo "$i"
#for each layer name and neuron index
#iterate over the .pb files
        for (( i=1; i<=number_of_pb_files; i++ ))
        do
            #echo $i
            python vis_neuron.py --MODEL_PATH=$i.pb --LAYER=$layer --NEURON_INDEX=$neuron_index
        done
        echo "merge.py-start"
        python merge.py --column=1 --name=merged_steps${layer////-}$neuron_index
        echo "merge.py-end"
        mv merged_steps${layer////-}$neuron_index.png $savefolder/
        rm *.png
    done

    

done < "$input"
