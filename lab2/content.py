# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 2: k-NN and Naive Bayes
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------


from __future__ import division

import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: set of objects that are going to be compared (size: N1xD)
    :param X_train: set of objects compared against param X (size N2xD)
    Functions calculates Hamming distances between all objects from X and all object from X_train.
    Resulting distances are returned as matrix.
    :return: Matrix of distances between objects X and X_train (size: N1xN2)
    """

    x = X.toarray().astype(int)
    x_trainT = np.transpose(X_train.toarray()).astype(int)
    return x.shape[1] - (x @ x_trainT) - (1 - x) @ (1 - x_trainT)
    #
    # mat = dok_matrix((X.shape[0], X_train.shape[0]))
    #
    # for ida, a in enumerate(X):
    #     for idb, b in enumerate(X_train):
    #         mat[ida, idb] = ((a.todense() ^ b.todense()) == 1).sum()
    #
    # return mat.todense()


def sort_train_labels_knn(Dist, y):
    """
    Function sorts labels of training data y accordingly to probabilities stored in matrix Dist.
    Function returns matrix N1xN2. In each row there are sorted data labels y accordingly to corresponding row of matrix Dist.
    :param Dist: Distance matrix between objects X and X_train (size: N1xN2)
    :param y: vector of labels (N2 elements)
    :return: Matrix of sorted class labels (use mergesort algorithm)
    """
    sorter = Dist.argsort(kind="mergesort")
    return y[sorter]


def p_y_x_knn(y, k):
    """
    Function calculates conditional probability p(y|x) for
    all classes and all objects from test set using KNN classifier
    :param y: matrix of sorted labels for training set (size: N1xN2)
    :param k: number of nearest neighbours
    :return: matrix of probabilities for objects X
    """

    sliced_matrix = y[:, :k]
    classes_amount = len(np.unique(y[0, :]))

    res = np.zeros((y.shape[0], classes_amount))
    for idr, row in enumerate(sliced_matrix):
        for x in row:
            res[idr, x] += 1
    return res / k


def classification_error(p_y_x, y_true):
    """
    Function calculates classification error
    :param p_y_x: matrix of predicted probabilities
    :param y_true: set of ground truth labels (size: 1xN).
    Each row of matrix represents distribution p(y|x)
    :return: classification error
    """

    classifications = p_y_x.argsort(kind="mergesort", axis=1)[:, -1:].squeeze()
    return (y_true != classifications).sum() / y_true.shape[0]


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: validation data (size: N1xD)
    :param Xtrain: training data (size: N2xD)
    :param yval: class labels for validation data (size: 1xN1)
    :param ytrain: class labels for training data (size: 1xN2)
    :param k_values: values of parameter k that must be evaluated
    :return: function performs model selection with knn and returns tuple (best_error,best_k,errors), where best_error is the lowest
    error, best_k - value of k parameter that corresponds to the lowest error, errors - list of error values for
    subsequent values of k (elements of k_values)
    """

    distances = hamming_distance(Xval, Xtrain)
    training = sort_train_labels_knn(distances, ytrain)

    def f(k):
        prob_train = p_y_x_knn(training, k)
        return classification_error(prob_train, yval)

    errors = list(map(f, k_values))

    min_pos = np.argmin(errors)
    best_error = errors[min_pos]
    best_k = k_values[min_pos]

    return best_error, best_k, errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: labels for training data (size: 1xN)
    :return: function calculates distribution a priori p(y) and returns p_y - vector of a priori probabilities (size: 1xM)
    """

    amounts = {}
    for y in ytrain:
        amounts[y] = 1 + amounts.get(y, 0)

    res = np.ndarray((len(amounts),))
    for idx, val in amounts.items():
        res[idx] = val

    return res / res.sum()


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: training data (size: NxD)
    :param ytrain: class labels for training data (size: 1xN)
    :param a: parameter a of Beta distribution
    :param b: parameter b of Beta distribution
    :return: Function calculates probality p(x|y), assuming that x is binary variable and elements of x 
    are independent from each other. Function returns matrix p_x_y (size: MxD).
    """

    Xtrain = Xtrain.toarray()
    up_factor = a - 1.0
    down_factor = a + b - 2.0

    def f(k, d):
        I_yn_k = ((ytrain + 1) == k + 1).astype(bool)
        I_xnd_1 = (Xtrain[:, d] == 1).astype(bool)
        up = up_factor + np.count_nonzero(I_yn_k & I_xnd_1)
        down = down_factor + np.count_nonzero(I_yn_k)
        return up / down

    return np.fromfunction(np.vectorize(f), shape=(4, Xtrain.shape[1]), dtype=int)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: vector of a priori probabilities (size: 1xM)
    :param p_x_1_y: probability distribution p(x=1|y) (matrix, size: MxD)
    :param X: data for probability estimation, matrix (matrix, size: NxD)
    :return: function calculates probability distribution p(y|x) for each class with the use of Naive Bayes classifier.
    Function returns matrix p_y_x (size: NxM).
    """

    X = X.toarray()
    p_x_1_y_rev = 1 - p_x_1_y
    X_rev = 1 - X
    res = []
    for i in range(X.shape[0]):
        success = p_x_1_y ** X[i,]
        fail = p_x_1_y_rev ** X_rev[i,]
        a = np.prod(success * fail, axis=1) * p_y
        sum = np.sum(a)
        res.append(a / sum)
    return np.array(res)


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: training set (size: N2xD)
    :param Xval: validation set (size: N1xD)
    :param ytrain: class labels for training data (size: 1xN2)
    :param yval: class labels for validation data (size: 1xN1)
    :param a_values: list of parameters a (Beta distribution)
    :param b_values: list of parameters b (Beta distribution)
    :return: Function performs a model selection for Naive Bayes. It selects the best values of a i b parameters.
    Function returns tuple (error_best, best_a, best_b, errors) where best_error is the lowest error,
    best_a - a parameter that corresponds to the lowest error, best_b - b parameter that corresponds to the lowest error,
    errors - matrix of errors all pairs (a,b)
    """

    A = len(a_values)
    B = len(b_values)

    p_y = estimate_a_priori_nb(ytrain)

    def f(cord1, cord2):
        p_x_y_nb = estimate_p_x_y_nb(Xtrain, ytrain, a_values[cord1], b_values[cord2])
        p_y_x = p_y_x_nb(p_y, p_x_y_nb, Xval)
        err = classification_error(p_y_x, yval)
        return err

    errors = np.fromfunction(np.vectorize(f), shape=(A, B), dtype=int)

    min = np.argmin(errors)
    minA = min // A
    minB = min % A
    return errors[minA, minB], a_values[minA], b_values[minB], errors
