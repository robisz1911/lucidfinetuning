1) download the selected dataset you wanna to finetune on
   copy the dateset to this folder

links:
   celeba dataset: https://github.com/robisz1911/celeba_dataset
   animals dataset: https://github.com/robisz1911/LUCID_RESULTS/tree/master/DATASETS/animals
   flowers17 dataset: https://github.com/robisz1911/LUCID_RESULTS/tree/master/DATASETS/flowers17

2) modify the_script.sh

parameters:

   training= True -> finetune to the selected dataset from imagenet weights
             False -> If you've already finetuned a model
   pb_gen  = True -> for visualization, you have to generate .pb from .ckpt files
             False -> You've done this previously

   save_checkpoints = how many checkpoints are you intrested in
      True -> at the beggining of the training, after every epoch a checkpoint 'll be saved,
              further the training, the less checkpoints are saved
      False -> only two checkpoints are made, at the initial state, and at the end of the last epoch
   
   visualize_all_steps
      True -> visualize the whole finetuning process through the saved checkpoints
              create a visualization in every checkpoint, and merge them in order to one picture
              example: merged_stepsMixed_3c_Branch_3_b_1x1_act-Relu41.png

   visualize_only_the_last_epoch
      True -> visualizations are created at the initial state and after the last epoch only,
              also merge them into one picture
   
   if visualize_all_steps AND visualize_only_the_last_epoch are both set to False -> no visualization process
   
   nb_epoch = Number of epoch to finetune for

   dataset = flowers17/celeba/animals

   steps_per_epoch = number of training steps in each epoch
      if equals to 1 -> it'll finetune on only one batch in each epoch

   batch_size = batch size

   cutoff = first 'cutoff' layers are set to non trainable

   input = "something.txt"
      "layername" "list of neurons comma separated"
      example format can be found in: "layers_neuron_list.txt"
      set "neuron_list_for_neuron_catalog.txt"

3) run the_script.sh (./the_script.sh)