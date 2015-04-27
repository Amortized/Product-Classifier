"""

__author__ = 'amortized'
"""

import pandas as pd;
import numpy  as np;
from sklearn import preprocessing, cross_validation;

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


def crossvalidate(train_Y, K):
    skf = cross_validation.StratifiedKFold(train_Y, K, shuffle=True)
    for train_index, test_index in skf:
        class_dist = np.histogram(train_Y[train_index], bins=[0,1,2,3,4,5,6,7,8])[0];
        class_dist = class_dist / float(sum(class_dist));
        print(class_dist)



if __name__ == '__main__':
    lbl_enc, test_ids, train_X, train_Y, test_X = readData("./data/train.csv","./data/test.csv");
    crossvalidate(train_Y, 10)