#!/bin/bash


# visualize chosen layer from .pb files in 'training_x'

######  NOTE : before running the script, delete 'pictures_x' folder ######

mkdir pictures_x
cp merge.py pictures_x

rows=100                                              # the final image will have this many rows
layer=Mixed_4c_Branch_3_b_1x1_act/Relu

#sed -i 's/COLUMNS = .*/COLUMNS = 15/' vis.py       # SET : number of neurons visualized       |THESE TWO-      |
#sed -i 's/columns = .*/columns = 15/' merge.py     # SET : number of neurons visualized       |MUST BE THE SAME|

# for i in {1..46..5}
for (( i = 1; i <= $rows; i++ ))                    
do
                                                    # visualize, then copy .png to 'pictures_x'
						    # set the layer after grep below
    cat googlenet-node-names | grep $layer | python vis.py training_x/$i.pb - $i
    mv $i.png pictures_x

done
                                                    # merge all images in 'pictures_x' - merged.png
						    
cd pictures_x
python merge.py
