import numpy as np
import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import time

def classify_trees(filename, classifierdata, path):
    tree_array = np.load("{}{}{}".format(path,  "/Data/Temp/", filename))
    training_samples_mask = np.array([False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      True, True, False])
    X = tree_array[:, training_samples_mask]
    X = MinMaxScaler().fit_transform(X)

    clf = joblib.load("{}{}{}".format(path, "/Data/Rotterdam/Training Data/TreeTypeFiles/Brand New/", classifierdata))

    output = clf.predict(X)

    scorelist = []
    for i in clf.predict_proba(X):
        scorelist.append(round(max(i), 2))

    output_array = np.column_stack((tree_array, output, scorelist))

    np.save("{}{}{}{}".format(path, "/Data/Temp/Brand New/", filename[:-4], "_Classified.npy"), output_array)
    return output_array

if __name__ == '__main__':
    """GLOBAL VARIABLES"""
    path = os.getcwd()

    classifierdata = "NN_Clades.pkl"
    filename = "Delfshaven_test.npy"


    input_param_filename = "Parameters to classify on when a training set is determined.npy"

    starttime = time.clock()

    classify_trees(filename, classifierdata, path)

    print "Classifying treetypes took:", time.clock() - starttime, "seconds."