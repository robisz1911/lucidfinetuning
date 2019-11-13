#!/bin/bash


input="layers_name_channel_concat.txt"


while IFS= read -r line
do

    vars=( $line )
    layer=${vars[0]}
    n_neurons=${vars[1]}-1

    echo $layer
    echo $n_neurons

    dir=${layer////-}
    cd $dir
    
    mergeddir=${dir}_merged

    mkdir $mergeddir

# for number of neurons in layer
    for (( i=0; i<=n_neurons; i++ ))
    do
        #echo $i
	
	if stat -t ${dir}_${i}_* >/dev/null 2>&1
	then
		python ../merge_neuron.py $layer $i

		mv ${dir}_${i}.png $mergeddir
	fi
	
    done

    cd ..

done < "$input"
