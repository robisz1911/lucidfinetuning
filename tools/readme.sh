
mkdir gathered
time find . | grep "celeba_neuron_catalog.*/Mixed.*/.*.pb.png" | while read f ; do cp $f gathered ; done


# activations.py

# main_convert() as main()
time echo gathered/* | tr ' ' '\n' | grep googlenet_default | python activations.py lucid-default-imagenet-256.npy > sorted.txt
# -> lucid-default-imagenet-256.npy
time echo gathered/* | tr ' ' '\n' | grep googlenet_finetuned | python activations.py lucid-celeba-256.npy > sorted-celeba.txt
# -> lucid-celeba-256.npy


# condense.py collects mean RGB, will be able to do FFT analysis.
time ls gathered/ | sed "s/^/gathered\//" | python condense.py > cout


# vis.py creates t-SNE vis based on activation pattern similarities.