
mkdir gathered
time find . | grep "celeba_neuron_catalog.*/Mixed.*/.*.pb.png" | while read f ; do cp $f gathered ; done


# activations.py

# main_convert_old() as main(), UPDATE: this is now obsoleted, see main_convert()
time echo gathered/* | tr ' ' '\n' | grep googlenet_default | python activations.py lucid-default-imagenet-256.npy > sorted.txt
# -> lucid-default-imagenet-256.npy
time echo gathered/* | tr ' ' '\n' | grep googlenet_finetuned | python activations.py lucid-celeba-256.npy > sorted-celeba.txt
# -> lucid-celeba-256.npy


# condense.py collects mean RGB, will be able to do FFT analysis.
time ls gathered/ | sed "s/^/gathered\//" | python condense.py > cout


# vis.py creates t-SNE vis based on activation pattern similarities.


#######
# completing https://users.renyi.hu/~daniel/lucid/

# merging the new-old default and celeba layers:

# geforce1
ls /home/robisz1911/10.25./lucidfinetuning/neuron_catalog_1-10/layers_name_channel_relu[1-9]*.txt | while read f ; do dir=neuron_catalog`echo $f | sed "s/.*relu/_/" | cut -f1 -d'.' ` ; cat $f | sed "s/act\/Relu/act-Relu/" | sed "s/^/$dir /" ; done | sed "s/neuron_catalog_11-20/neuron_catalog_1-20/" | sed "s/neuron_catalog_1-10/neuron_catalog_1-20/" > layers_dirs.txt
scp -q -r layers_dirs.txt renyi.hu:./www/tmp/lucid/catalogs/
scp -q -r /home/robisz1911/10.25./lucidfinetuning/neuron_catalog_1-10 renyi.hu:./www/tmp/lucid/catalogs/
scp -q -r /home/robisz1911/10.25./lucidfinetuning/neuron_catalog_11-20 renyi.hu:./www/tmp/lucid/catalogs/
# beware the underscore:
scp -q -r /home/robisz1911/10.25./lucidfinetuning/celeba_neuron_catalog1-20 renyi.hu:./www/tmp/lucid/catalogs/celeba_neuron_catalog_1-20


# hexagon
cd ~/www/tmp/lucid/catalogs
python layer_txt_to_js.py < layers_dirs.txt > layers_dirs.js
cp layers_dirs.js ~/www/lucid/layers.js
# copying the defaults next to their celeba pair:
# (just the defaults, their directories have finetuned images but we skip them.)
echo neuron_catalog_1-10/* neuron_catalog_11-20/* | tr ' ' '\n' | grep "/Conv\|/Mixed" | while read d ; do dd=`echo $d | sed "s/neuron_catalog_11-20/neuron_catalog_1-20/" | sed "s/neuron_catalog_1-10/neuron_catalog_1-20/"` ; cp $d/*_default.pb.png celeba_$dd ; done | less


#######
# creating the npy image datasets from the above:

# main_convert() is the entry
time python activations.py celeba default lucid-celeba-full-256.npy < layers_dirs.txt > lucid-celeba-full-256.txt
time python activations.py celeba default lucid-default-imagenet-full-256.npy < layers_dirs.txt > lucid-default-imagenet-full-256.txt
# ouch, there are 4 default images missing. copied them to localhost and hexagon celeba_neuron_catalog_21-30
