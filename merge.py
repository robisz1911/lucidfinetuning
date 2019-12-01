# iterate over png files in the current folder, and merge them into one png (merged.png) 
# PARAMETER :
#                       column    :    SAME as the COLUMN in vis.py ( the number of neurons visualized )
# NOTE : if you change the size of the generated images(in vis.py), change height and width here too!
from PIL import Image
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--column", help='number of neurons in each layer', type=int)
parser.add_argument("--name", help='name of the saved file', type=str, default="merged")

FLAGS = parser.parse_args()

name = FLAGS.name
column = FLAGS.column


def merge():
    number_of_pictures = 0
    for filename in os.listdir():
        if filename.endswith(".png"):
            x = filename.zfill(40)
            os.rename(filename,x)
            number_of_pictures += 1
    print("-----------------------------------------------------")
    print(number_of_pictures)        
    height = 128
    #column = 5 # must be the same as COLUMN in vis.py ( the number of neurons we wanna to visualize )
#    column_arg = sys.argv[1:]
#    print(column_arg)
#    column = int(column_arg)
    width = 128*column
    merged = Image.new('RGB', (width, height*number_of_pictures))
    iterator = 0
    for filename in sorted(os.listdir()):
        if filename.endswith(".png"):
            print(filename)
            y = Image.open(filename)
            merged.paste(y, (0, iterator*height))
            iterator += 1
    return merged

merge().save(name+'.png')
