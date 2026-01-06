# mito_redox_analysis
This repository contains analysis of Mitochondrial Networks and attempts to use their structural properties to predict the oxidation function across cancerous and non cancerous experimental groups. This work was done under the supervision of Dr. Kasturi Mitra with the support of Mphasis Applied AI and Tech Lab.

Currently documented is a Random Forest pipeline (directory: rl_dl) which inputs a Network's structural features and trains a random forest Model on it to predict the Regression variable. This process is carried out on a dataset which pools 10 different experimental conditions for the mitochondrial network. The direcotry has the following file structure:
#### rl_dl
- data_loader.py : Loads the initial data and performs preprocessing tasks such as dropping duplicates based on a key element to reduce the scale of the task, log normalising all length based features for a "more" Gaussian distribution, dropping NaNs and Infs and filtering outliers based on target variable distribution analysis.
- eval.py : Contains functions for train/test split, creating model instance, performing 5 Fold Cross Validation on the training set, and finally, fitting the model to the full training set and testing. Main criterion for model performance is R2 score to measure explained variance.
- tune_hyperparams.py : Performes a Bayes Search Cross Validation (from SKOPT library) to find the optimal hyperparameters for the Random Forest Regressor.
- model_analysis.py : Plots feature importance of the RF Model and also plots a Regression Plot of Actual Vs Predicted values with a 45 Degree reference line.
- dataset_size_reduction_exp.py : An experiment where the model performance is tracked with reducing training set size in fractions of 10%, from all the training data to only 10% of it to pinpoint the minimal data points needed for the model to learn without degrading performance.
- execution.py : Contains the main executable for this pipeline, incorporating the above functions.
