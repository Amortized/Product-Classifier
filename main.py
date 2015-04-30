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


def train_model(features, label, params, K, class1):

    skf = cross_validation.StratifiedKFold(label, K, shuffle=True);

    total_log_loss  = 0.0;
    count           = 0;

    for train_index, validation_index in skf:

        X_train, X_validation = features[train_index], features[validation_index];
        Y_train, Y_validation = label[train_index], label[validation_index];

        estimator             = GradientBoostingClassifier(**params)

        estimator.fit(X_train, Y_train);

        current_log_loss      = log_loss(Y_validation, estimator.predict_proba(X_validation));

        total_log_loss       += current_log_loss;
        count                += 1;

    #Average across all samples
    avg_log_loss              = total_log_loss / float(count);
    print("Avg Log Loss for Classifier  " +  str(class1) + " is " + str(avg_log_loss));

    del features;
    del label;

    return  (params, avg_log_loss);

def generateParams():
    # Set the parameters by cross-validation
    paramaters_grid    = {'max_depth': [5], 'min_samples_split' : [5],  'min_samples_leaf' : [5], 'n_estimators' : [1000]};

    paramaters_search  = list(ParameterGrid(paramaters_grid));

    parameters_to_try  = [];
    for ps in paramaters_search:
        params           = {'max_features' : 'sqrt', 'learning_rate' : 0.05};
        for param in ps.keys():
            params[str(param)] = ps[param];
        parameters_to_try.append(copy.copy(params));

    return parameters_to_try;     

def train_model_wrapper(args):
   return train_model(*args);


def buildBestBinaryCLassifier(features, label, class1):
    """
       label : 0 or 1
    """

    K = 2;
    print("Building a binary classifier between " + str(class1) + " and others ");
    parameters_to_try = generateParams();

    #Contruct parameters as s list
    models_to_try     = [ (copy.copy(features), copy.copy(label), parameters_to_try[i], K, class1 ) for i in range(0, len(parameters_to_try)) ];
    
    #Create a Thread pool.
    pool              = Pool(8);
    results           = pool.map( train_model_wrapper, models_to_try );

    pool.close();
    pool.join();
    

    best_params       = None;
    best_log_loss     = sys.float_info.max;
    for i in range(0, len(results)):
      if results[i][1] < best_log_loss:
         best_log_loss   = copy.copy(results[i][1]);
         best_params     = copy.copy(results[i][0]);

    print("Best Params : " + str(best_params));
    print("Best RMSE :   " + str(best_log_loss));

    estimator             = GradientBoostingClassifier(**best_params)
    estimator.fit(features, label);

    del results;
    del features;
    del label;
    del models_to_try;
    
    return estimator;


def do_one_vs_all(X_train, Y_train):
    estimators = dict();
    labels = np.unique(Y_train);
    for label in labels:
        #Get all the samples with this label
        class_1_dataset = np.array([X_train[i] for i in range(0, len(Y_train)) if Y_train[i] == label]);
        class_0_dataset = np.array([X_train[i] for i in range(0, len(Y_train)) if Y_train[i] != label]);
        
        #Randomly downsample class_0 dataset without replacement
        random.shuffle(class_0_dataset);
        #class_0_dataset = class_0_dataset[np.array(np.random.choice(len(class_0_dataset), len(class_1_dataset), replace=False))]; 

        #Prepare labels
        class_1_Y       = np.array([1 for i in range(0, len(class_1_dataset))]);
        class_0_Y       = np.array([0 for i in range(0, len(class_0_dataset))]);

        class_0_1_X     = np.concatenate((class_1_dataset, class_0_dataset));
        class_0_1_Y     = np.concatenate((class_1_Y, class_0_Y));

        assert(len(class_0_1_Y) == len(class_0_1_X))

        estimator = buildBestBinaryCLassifier(copy.copy(class_0_1_X), copy.copy(class_0_1_Y), label);

        estimators[label] = estimator; #We store one vs rest estimator for this label;

    return estimators;
   

def predict(X_validation, estimators):
    predict_probabilites = [];
    print("Total no of predictions : " + str(len(X_validation)));
    count = 0;
    for X in X_validation:
       if count % 100 == 0:
         #print("Predicted : " + str(count));
         pass;
       count += 1;
       #Probabilities are in order as class 0 .. class 8
       probabilites = [estimators[label].predict_proba(X)[0][1] for label in sorted(estimators.keys())]; 
       #Normalize 
       probabilites = [prob/float(sum(probabilites)) for prob in probabilites];
       predict_probabilites.append(probabilites);

    return np.array(predict_probabilites);



def calculate_loss(X_validation, Y_validation, estimators):
    return log_loss(Y_validation, predict(X_validation, estimators));


def write_test(estimators, test_ids, test_X, lbl_enc):
    predict_probabilites = predict(test_X, estimators);
    f = open("./data/submission.csv", "w");
    f.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n");
    for i in range(0, len(predict_probabilites)):
        f.write(str(test_ids[i]) + "," + str(predict_probabilites[i][0])  + "," + str(predict_probabilites[i][1])  + "," + str(predict_probabilites[i][2])  + "," + str(predict_probabilites[i][3])  + "," + str(predict_probabilites[i][4])  + "," + str(predict_probabilites[i][5])  + "," + str(predict_probabilites[i][6])  + "," + str(predict_probabilites[i][7])  + "," + str(predict_probabilites[i][8]) + "\n"); 
    f.close();


def build(features, label):

    X_train, X_validation, Y_train, Y_validation = train_test_split(features, label, test_size=0.20, random_state=100);
    estimators       = do_one_vs_all(copy.copy(X_train), copy.copy(Y_train));
    total_log_loss   = calculate_loss(X_validation, Y_validation, estimators);
    print("Log Loss :" + str(total_log_loss));
    return estimators;


if __name__ == '__main__':
    warnings.filterwarnings("ignore");
    lbl_enc, test_ids, train_X, train_Y, test_X = readData("./data/train.csv","./data/test.csv");
    estimators = build(train_X, train_Y);
    write_test(estimators, test_ids, test_X, lbl_enc);
