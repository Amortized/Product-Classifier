"""

__author__ = 'amortized'
"""

import pandas as pd;
import numpy  as np;
from sklearn import preprocessing, cross_validation;
from sklearn.metrics import log_loss;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import GradientBoostingClassifier;
from sklearn.grid_search import ParameterGrid;
from multiprocessing import Pool;
import copy;
import random;
import sys;
import warnings;
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix;
import matplotlib.pyplot as plt;
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import copy

def readData(train_f, test_f):
    train   = pd.read_csv(train_f);
    test    = pd.read_csv(test_f);

    #Remove the id's and extract label
    train_Y  = train.target.values;
    test_ids = test.id.values;

    #Remove the id and label
    train.drop(["id","target"], inplace=True, axis=1);
    train_X = np.array(train);

    test.drop(["id"], inplace=True, axis=1);
    test_X = np.array(test);

    #Encode the labels
    lbl_enc = preprocessing.LabelEncoder();
    train_Y = lbl_enc.fit_transform(train_Y)

    return lbl_enc, test_ids, train_X, train_Y, test_X;


def plot_confusion_matrix(y, y_prediction, title='Normalized Confusion matrix', cmap=plt.cm.Blues):
    labels = list(set(y.tolist()));
    labels.sort();

    y_prediction = np.array([prob.tolist().index(max(prob))  for prob in y_prediction]);

    cm = confusion_matrix(y, y_prediction);
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig("confusion_matrix.png")


def generateParams():
    # Set the parameters by cross-validation
    paramaters_grid    = {'eta': [0.05], 'min_child_weight' : [4],  'colsample_bytree' : [0.8], 'subsample' : [0.90], 'gamma' : [0], 'max_depth' : [12]};

    paramaters_search  = list(ParameterGrid(paramaters_grid));

    parameters_to_try  = [];
    for ps in paramaters_search:
        params           = {'eval_metric' : 'mlogloss', 'objective' : 'multi:softprob', 'num_class' : 9, 'nthread' : 8};
        for param in ps.keys():
            params[str(param)] = ps[param];
        parameters_to_try.append(copy.copy(params));

    return parameters_to_try;     


def build(features, label):
    X_train, X_validation, Y_train, Y_validation = train_test_split(features, label, test_size=0.20, random_state=100);

    #Load Data
    dtrain      = xgb.DMatrix( X_train, label=Y_train);
    dvalidation = xgb.DMatrix( X_validation, label=Y_validation);

    parameters_to_try = generateParams();
    for i in range(0, len(parameters_to_try)):
        param     = parameters_to_try[i]
        #Train a Model
        evallist  = [(dtrain,'train'), (dvalidation,'eval')]
        num_round = 1000
        bst       = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)

        print("Best Validation Score " + str(bst.best_score));
        bst.save_model('./data/best_model.model')



def write_test(bst, test_ids, test_X):
    ndp    = len(test_X)
    test_X = xgb.DMatrix(test_X);

    predict_probabilites = bst.predict( test_X ,ntree_limit=bst.best_iteration).reshape( ndp, 9 )
    f = open("./data/submission.csv", "w");
    f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n");
    for i in range(0, len(predict_probabilites)):
        f.write(str(test_ids[i]) + "," + str(predict_probabilites[i][0])  + "," + str(predict_probabilites[i][1])  + "," + str(predict_probabilites[i][2])  + "," + str(predict_probabilites[i][3])  + "," + str(predict_probabilites[i][4])  + "," + str(predict_probabilites[i][5])  + "," + str(predict_probabilites[i][6])  + "," + str(predict_probabilites[i][7])  + "," + str(predict_probabilites[i][8]) + "\n"); 
    f.close();



if __name__ == '__main__':
    warnings.filterwarnings("ignore");
    lbl_enc, test_ids, train_X, train_Y, test_X = readData("./data/train.csv","./data/test.csv");
    #build(train_X, train_Y);

    #Load model
    bst = xgb.Booster({'nthread':8})
    bst.load_model('./data/best_model.model')

    write_test(bst, test_ids, test_X)


