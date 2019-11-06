#!/bin/bash
# VISUALIZE FINETUNING STEP BY STEP

# VISUALIZE 'layer' from .pb files in 'foldername' and concatenate these pictures into one with merge.py

# NOTE : SET THESE PARAMETERS BELOW:

rows=2  #number of pb files
columns=1  #number of neurons
folder_for_pictures=lasttest_pics                    # where you save the pictures
folder_where_pb_files_are=lasttest                              # from where you read .pb files
layer=Mixed_4c_Branch_3_b_1x1_act/Relu

mkdir $folder_for_pictures
cp merge.py $folder_for_pictures

for (( i = 1; i <= $rows; i++ ))                    
do
    cat googlenet-node-names | grep $layer | python vis.py $folder_where_pb_files_are/$i.pb - $i $columns
    mv $i.png $folder_for_pictures
done

cd $folder_for_pictures
python merge.py --column=$columns
