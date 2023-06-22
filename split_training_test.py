from utility import parse
import pandas as pd
from utility import preprocessing_text, features_selection, format_missing_values
from sklearn.model_selection import StratifiedShuffleSplit
import os

RANDOM_STATE = 7

# read original dataset and split in features and class
df = pd.DataFrame(list(parse('./amz_products_small.jsonl.gz')))
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# training and testing datasets
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
for train_index, test_index in stratified_split.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

# save the files in the proper folder
X_train.to_csv('Data/X_train.csv', index=False)
X_test.to_csv('Data/X_test.csv', index=False)
y_train.to_csv('Data/y_train.csv', index=False)
y_test.to_csv('Data/y_test.csv', index=False)

X_train_preprocessed = features_selection(format_missing_values(X_train))
X_train_preprocessed['feature'] = X_train_preprocessed['feature'].apply(preprocessing_text)
X_train_preprocessed.to_csv('Data/X_train_preprocessed.csv', index=False)

X_test_preprocessed = features_selection(format_missing_values(X_test))
X_test_preprocessed['feature'] = X_test_preprocessed['feature'].apply(preprocessing_text)
X_test_preprocessed.to_csv('Data/X_test_preprocessed.csv', index=False)

# save few rows of the dataset to test the inference of the model deployed
(df.sample(n=5, random_state=RANDOM_STATE)).to_csv('Data/inference.csv', index=False)
