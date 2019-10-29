#!/bin/bash

# VISUALIZE 'layer' from .pb files in 'foldername' and concatenate these pictures into one with merge.py

######  NOTE : before running the script, rename foldername to a non existing foldername ######

#sed -i 's/COLUMNS = .*/COLUMNS = 15/' vis.py       # SET : number of neurons visualized       |THESE TWO-      |
#sed -i 's/columns = .*/columns = 15/' merge.py     # SET : number of neurons visualized       |MUST BE THE SAME|

rows=5  #number of pb files
columns=5  #number of neurons
folder_for_pictures=aaa_pics
foldername=aaa_pbs
layer=Mixed_4c_Branch_3_b_1x1_act/Relu

mkdir $folder_for_pictures
cp merge.py $folder_for_pictures

for (( i = 1; i <= $rows; i++ ))                    
do

     #python vis.py --MODEL_PATH=googlenetLucid.pb --LAYER = cat googlenet-node-names | grep Branch_2_b_3x3_act/Relu --OUTPUT_PREFIX=test --column=5

    cat googlenet-node-names | grep $layer | python vis.py $foldername/$i.pb - $i $columns
    mv $i.png $folder_for_pictures

done
						    
cd $folder_for_pictures
python merge.py --column=$columns
