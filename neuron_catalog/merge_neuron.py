import sys
import os
from PIL import Image

LAYER, NEURON_INDEX = sys.argv[1:]

LAYER = LAYER.replace("/","-")
pixels = 256

number_of_pictures = 0
for filename in os.listdir():
    if LAYER+"_"+NEURON_INDEX+"_" in filename:
        number_of_pictures += 1


merged = Image.new('RGB', (pixels, pixels*number_of_pictures))
iterator = 0
for filename in sorted(os.listdir()):
    if LAYER+"_"+NEURON_INDEX+"_" in filename:
        print(filename)
        y = Image.open(filename)
        merged.paste(y, (0, iterator*pixels))
        iterator += 1

out_filename = LAYER+"_"+NEURON_INDEX+".png"

merged.save(out_filename)

print(out_filename + " saved")
