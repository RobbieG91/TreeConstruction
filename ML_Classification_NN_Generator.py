import numpy as np
import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.model_selection import KFold
import time

# To view full arrays
np.set_printoptions(threshold=sys.maxsize)

def read_data(path, train_trees, treetypes, trees):
    """MLP Classifier uses X to predict y. X being the features (tree parameters) used to predict tree-types y.
    The 9 parameters Tree modelling parameters are the following:
    1. Point Count (might remove later)
    2. Crown Base
    3. Periphery Height
    4. Periphery Radius
    5. Lower Periphery Height
    6. Lower Periphery Radius
    7. Higher Periphery Height
    8. Higher Periphery Radius
    9. Tree Top

    With these parameters, several features have been set up that make for a stronger NN:
    Ratios:

    Height
    10. Higher Periphery / Lower Periphery
    11. Higher Periphery / Periphery
    12. Periphery / Lower Periphery
    13. Tree Top / Crown Base

    Radius
    14. Higher Periphery / Lower Periphery
    15. Higher Periphery / Periphery
    16. Periphery / Lower Periphery

    Extra
    17. Ratio Periphery Height / Periphery Radius
    18. Average Intensity
    19. Average Number of Returns

    Hidden layers can be used to assign different weights to every feature, however there will be no weights assigned
    to the different features, as for me it's unsure which tree parameters would be more important than others.

    The types that are used are the types found in the training dataset from Rotterdam 3D. We will be using the
    latin names."""

    tree_array = np.load("{}{}{}".format(path, "/Data/Rotterdam/Training Data/New Selected Features/", train_trees))
    tree_types = np.genfromtxt("{}{}{}".format(path, "/Data/Rotterdam/Training Data/New Selected Features/", treetypes), delimiter='\t', skip_header=1, dtype='string')
    tree_array_todo = np.load("{}{}{}".format(path, "/Data/Temp/", trees))

    keep_indices = []
    for tree_id in np.nditer(tree_types[:, 0]):
        keep_indices.extend(np.where(float(tree_id) == tree_array[:, 0])[0])
    X_full = tree_array[keep_indices]

    """Eight best features"""
    # training_samples_mask = np.array([False, False, False, False, False, False, False, False, False, True,
    #                                   False, False, False, True, True, False, True, True, True, False,
    #                                   True, False, True])
    """Six best features"""
    # training_samples_mask = np.array([False, False, False, False, False, False, False, False, False, True,
    #                                   False, False, False, True, True, False, False, True, False, False,
    #                                   True, False, True])

    """All features"""
    # training_samples_mask = np.array([True, True, True, True, True, True, True, True, True, True,
    #                                   True, True, True, True, True, True, True, True, True, True,
    #                                   True, True, True])

    """Only intensity, number of returns and crown base height"""
    # training_samples_mask = np.array([False, False, False, False, False, True, False, False, False, False,
    #                                   False, False, False, False, False, False, False, False, False, False,
    #                                   True, True, False])
    """Only Treen intensity and number of returns"""
    training_samples_mask = np.array([False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      True, True, False])
    """Genera"""
    # target_values_mask = np.array([False, True])
    """Tree Type Conifer vs non-conifer"""
    target_values_mask = np.array([False, False, True])

    X = X_full[:, training_samples_mask]
    X = MinMaxScaler().fit_transform(X)

    y = tree_types[:, target_values_mask]

    X_todo = tree_array_todo[:, training_samples_mask]

    X_todo = MinMaxScaler().fit_transform(X_todo)

    kf = KFold(n_splits=5)
    clf = MLPClassifier(max_iter=100,
                        alpha=1e-05,
                        early_stopping=True,
                        solver='adam',
                        random_state=1)

    for train_indices, test_indices in kf.split(X):
        clf.fit(X[train_indices], y[:, 0][train_indices])
        print(clf.score(X[test_indices], y[:, 0][test_indices]))
    """
    In case you want to run gridsearchCV for different parameters
    mlp = MLPClassifier()

    # parameter_space = {
    #     'max_iter': 200,
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['lbfgs', 'sgd', 'adam'],
    #     'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    parameter_space = {
        'max_iter': [100],
        'solver': ['adam', 'sgd'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        # 'learning_rate': ['invscaling'],  # Only used in solver sgd apparently,
        'early_stopping': [True],
        'random_state': [0, 1, 2]
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1)

    clf.fit(X, y[:, 0])

    # Best parameter set
    print('Best parameters found:\n', clf.best_params_)
    bestparams = clf.best_params_
    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    """

    # clf.fit(X, y[:, 0])
    # joblib.dump(clf, "{}{}{}".format(path, "/Data/Rotterdam/Training Data/TreeTypeFiles/Selected Features/", "NN_Clades.pkl"))
    # clf = joblib.load("{}{}".format(path, "/Data/Rotterdam/Training Data/TreeTypeFiles/Selected Features/NeuralNetwork1.pkl"))

    output = clf.predict(X_todo)

    print "iterations", clf.n_iter_
    print "Correct/Score: ", clf.score(X, y[:, 0])
    print "Number of types:", len(np.unique(y[:, 0]))
    print np.unique(y[:, 0])


    scorelist = []
    for i in clf.predict_proba(X_todo):
        scorelist.append(round(max(i), 2))

    output_array = np.column_stack((tree_array_todo, output, scorelist))

    np.save("{}{}{}".format(path, "/Data/Temp/ML_Arrays/", "Noordereiland_NewFeatures_ML"), output_array)

    return output_array

if __name__ == '__main__':
    """GLOBAL VARIABLES"""
    path = os.getcwd()

    training_param_filename = "Rotterdam_Combi.npy"
    target_values_filename = "UID_GENUS_TYPE.txt"

    input_param_filename = "Noordereiland_2std_100_Int_AllFeatures.npy"

    starttime = time.clock()
    somelist = []

    read_data(path, training_param_filename, target_values_filename, input_param_filename)
    print "Classifying treetypes took:", time.clock() - starttime, "seconds."