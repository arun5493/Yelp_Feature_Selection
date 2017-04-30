# Yelp_Feature_Selection

#### Dataset :
As the dataset was more than 100mb, could not upload it directly to the github.

Please find the dataset here:
https://drive.google.com/file/d/0B47NDHSkjCPyNzZTR09ETTRFZzA/view?usp=sharing

#### Directions of use :
- Clone this repository
- Download the dataset from the above mentioned google drive
- First run preprocess.py using the command: python preprocess.py
- The above file will generate a processed_data.csv. This file is being used by other R and python files.
- .py :
- models.r : Code for correlation, Ordinal regression and Mulitnomial regression.
- ann_ratings.r : Code for Aritificial neural network for rating prediction. 
- svm_lr_ratings.r : Code for Support vector Machine and Linear Regression


#### R Packages Required : 
- glmnet
- corrplot
- caret
- mlogit
- foreign
- MASS
- reshape2
- stats
- neuralnet
- nnet
- e1071

#### Python Libraries Required:
- json
- csv
- unicodedata
- ast
- tqdm
- operator
- numpy
- pandas
- scipy
- sklearn

