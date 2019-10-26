#!/bin/bash

# VISUALIZE 'layer' from .pb files in 'foldername' and concatenate these pictures into one with merge.py

######  NOTE : before running the script, rename foldername to a non existing foldername ######

#sed -i 's/COLUMNS = .*/COLUMNS = 15/' vis.py       # SET : number of neurons visualized       |THESE TWO-      |
#sed -i 's/columns = .*/columns = 15/' merge.py     # SET : number of neurons visualized       |MUST BE THE SAME|

rows=100   #number of pb files
columns=5  #number of neurons
foldername=picture_x
layer=Mixed_4c_Branch_3_b_1x1_act/Relu

mkdir $foldername
cp merge.py $foldername

for (( i = 1; i <= $rows; i++ ))                    
do

    cat googlenet-node-names | grep $layer | python vis.py $foldername/$i.pb - $i $columns
    mv $i.png $foldername

done
						    
cd $foldername
python merge.py $columns
