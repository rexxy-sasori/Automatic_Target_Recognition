This repository contains src code developed in python3 for training neural networks of Automatic 
Target Recognition. To run the code, run the command:

python main_train.py [YAML File Path]

This command will train a neural network model either from scratch or from a 
previously trained model. In the yaml folder, 2 YAML files have been prepared: one
for MSTAR and another one for SAMPLE. In both yamls files, the dataset_dir and result_dir
arguments need to be set properly. The training script will save the save the best model as
specified in the yaml file. It also offers an option to split the training data to create
a validation set. If use_validation is set to False, then the script will save the model with
either the best test error or test loss. 

When the training is completed, a model with the extension .pt.tar can be found in directory 
specified in result_dir. To evalute the model, run:

python main_eval.py [Model File Path]

The script will print the classification accuracy and draw the confusion matrix. Note: currently
the API only supports training on GPU and evaluating on GPU. If the feature of training on GPU and 
evaluating on CPU is desired, please submit the issue in wiki.

The dependency is listed in requirements.txt. # Automatic_Target_Recognition
