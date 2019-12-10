# Avila_Bible_kaggle

This is a project based on a [kaggle competition](https://www.kaggle.com/c/avila-bible-datamad1019)

In this competition we need to predict the scrib of a text.

My mains features were ALL OF THEM, so there's no cleaning or transformation


## Approaches
-----------------
In the approaches fold, there are 2 files in which ones I did the first approaches.

In first_look.py I try several models in order to have an aproximations of which one is the best for this data set. All the outputs were recorded in records.txt in the output folder.

I observed that the best model was Random Forest Regression (RFR), so I opted to do another approach with only RFR in silva_grid.py. Also the records of all approaches were saved in records_RFR.txt in the output folder.

## Predictions
----------------
Finally in the predictions folder there are 2 files:

- silva_pred.py : Here I predict the scriba with a Random Forest Classifier (RFC). This file generate a .csv in the output with the predictions.

- h2o_pred.py : H2O power trust her automl. This file generate a .csv in the output with the predictions.