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

The training and testing data is located in the input_data folder and the compressed version is in MSTAR_ROTATED.tar.gz. 

There are some pretrained models in the trained_models folder

The dependency is listed in requirements.txt.

If you are using this code and there is a resulting publication, please cite our work in ICASSP2020 as follows:

@INPROCEEDINGS{9054094, author={Dbouk, Hassan and Geng, Hanfei and Vineyard, Craig M. and Shanbhag, Naresh R.}, booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, title={Low-Complexity Fixed-Point Convolutional Neural Networks For Automatic Target Recognition}, year={2020}, volume={}, number={}, pages={1598-1602}, doi={10.1109/ICASSP40776.2020.9054094}}
 
