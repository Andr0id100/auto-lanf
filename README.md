# Advanced NLP Project

The different sections here describe the different components of the work done as part of this project.

## Pre-Processing
Contains a single file `preprocess.sh`. This is a bash file which upon its execution looks for two files, namely `corpus.en` and `corpus.de`. These files contain the parallel data for the model. The script performs the tasks of cleaning, tokenization, truecasing and extracting the vocab.  
The requirements for this are moses and subword-nmt.

## PyTorch Model
This folder contains the code for creating and training a model on the processed data created from the earlier part. The model is implemented in PyTorch. The translation quality from this model was not too good so following the lead of the authors of the paper, the same model configuration was also trained using Marian ToolKit.
Train Model:
```bash
python train.py
```

## Marian Model
The two folders, `without_ref` and `with_ref` contains the configuration files for the models that were trained without and with coreference data.  
In each folder, the `test_model.sh` file can be used to run inference on the model.  
Format:  
```bash
bash test_model.sh "This is a test"
```  
Before running this command, it is important to remember to place the model.npz file in the folder which had been removed for space constrains in the submisson.

The file `train_model.sh` describes the command to start the training.   
  

**Note:** All of these files make assumptions about the relative structures of the directories and the tools, etc. availble in them.