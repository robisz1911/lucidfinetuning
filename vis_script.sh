#!/bin/bash


# visualize chosen layer from .pb files in 'training_x'

######  NOTE : before running the script, rename foldername to a non existing foldername ######

#sed -i 's/COLUMNS = .*/COLUMNS = 15/' vis.py       # SET : number of neurons visualized       |THESE TWO-      |
#sed -i 's/columns = .*/columns = 15/' merge.py     # SET : number of neurons visualized       |MUST BE THE SAME|
rows=100
foldername = picture_x
layer=Mixed_4c_Branch_3_b_1x1_act/Relu
mkdir $foldername
cp merge.py $foldername

                                             # the final image will have this many rows



# for i in {1..46..5}
for (( i = 1; i <= $rows; i++ ))                    
do
                                                    # visualize, then copy .png to 'pictures_x'
						    # set the layer after grep below
    cat googlenet-node-names | grep $layer | python vis.py $foldername/$i.pb - $i
    mv $i.png $foldername

done
                                                    # merge all images in 'pictures_x' - merged.png
						    
cd $foldername
python merge.py
