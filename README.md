# QAring-ADL
Question Answering using NLP and Deep Learning
Script Information:
data.py : This script is to process the data by tokenizing, converting to word indexing, performing glove embedding and more.

bidaf.py : This is the bi directional attention flow model which is the main model. It takes two input, context and the question. It is the main model which we will train.

predict.py : This file is used to load the model to predict the answer to given questions.

main.py : This is the main entry point to the project. All commands are run from the main. It reads the data, calls and trains the model.

To run this:
The file structure should look like this: 
  - Data/ (download dataset from SQUAD) (we have preprocessed data but it is too huge, can send if required)
  - Scripts/ (download all the files from here)
  
Commands :
To train the model from scratch :
python main.py train none

To predict the results :
python main.py score best_bidaf.h5 
*note here best_bidaf.h5 is the best trained model*
