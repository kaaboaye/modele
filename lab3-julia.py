# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 3: Logistic regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------


import numpy as np


def sigmoid(x):
    '''
    :param x: input vector (size: Nx1)
    :return: vector of sigmoid function values calculated for x (size: Nx1)
    '''
    sigma = 1 / (1 + np.exp(-x))
    return sigma


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: model parameters (size: Mx1)
    :param x_train: training set features (size: NxM)
    :param y_train: training set labels (size: Nx1)
    :return: function returns tuple (val, grad), where val is a value of logistic function and grad is its gradient (calculated for parameters w)
    '''
    sigma = sigmoid(x_train @ w)
    value_nominator = (y_train * np.log(sigma)) + ((1 - y_train) * np.log(1 - sigma))
    value_denominator = y_train.shape[0] * -1
    value = np.sum(value_nominator/value_denominator)
    grad = - x_train.transpose() @ (y_train - sigma) / y_train.shape[0]
    return value, grad


def gradient_descent(obj_fun, w0, epochs, eta):

    '''
    :param obj_fun: objective function that is minimized. To call the function use expression "val,grad = obj_fun(w)".
    :param w0: starting point (size: Mx1)
    :param epochs: number of epochs / iterations of an algorithm
    :param eta: learning rate
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is vector of optimal model parameters and func_valus is vector of objective function values [epochs x 1], calculated for each epoch
	'''
    w = w0
    func_values = []
    v, grad = obj_fun(w0)
    for k in range(epochs):
        w = w - eta * grad
        val, grad = obj_fun(w)
        func_values.append(val)
    func_values = np.array(func_values).reshape(epochs, 1)
    return w, func_values


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
   
    """
	:param obj_fun: objective function that is minimized. To call the function use expression "val,grad = obj_fun(w,x,y)", 
	where x,y indicates mini-batches.
    :param x_train: training data (size: NxM)
    :param y_train: training data (size: Nx1)
    :param w0: starting point (size: Mx1)
    :param epochs: number of epochs
    :param eta: learning rate
    :param mini_batch: size of mini-batches
    :return: function optimizes obj_fun using stochastic gradient descent. It returns tuple (w,func_values),
    where w is vector of optimal value of model parameters and func_values is vector of objective function values [epochs x 1], calculated for each epoch.
    REMARK! Value of func_values is calculated for entire training set!
    """
    w = w0
    M = int(y_train.shape[0] / mini_batch)
    x_mb = np.vsplit(x_train, M) #x mini-batch
    y_mb = np.vsplit(y_train, M) #y mini-batch
    func_values = []
    for k in range(epochs):
        for x, y in zip(x_mb, y_mb):
            v, grad = obj_fun(w, x, y)
            w = w - eta * grad
        val, grad = obj_fun(w, x_train, y_train)
        func_values.append(val)
    func_values = np.array(func_values).reshape(epochs, 1)
    return w, func_values


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: model parameters (size: Mx1)
    :param x_train: training set - features (size: NxM)
    :param y_train: training set - labels (size: Nx1)
    :param regularization_lambda: regularization coefficient
    :return: function returns tuple (val, grad), where val is a value of logistic function with regularization l2,
    and grad is (calculated for model parameters w)
    '''
    val, grad = logistic_cost_function(w, x_train, y_train)
    w_0 = np.array(w)
    w_0[0] = 0
    val = val + regularization_lambda / 2 * np.linalg.norm(w_0) ** 2
    grad = grad + regularization_lambda * w_0
    return val, grad


def prediction(x, w, theta):
    '''
    :param x: observation matrix (size: NxM)
    :param w: vector of model parameters (size: Mx1)
    :param theta: classification threshold [0,1]
    :return: function calculates vector y (size: Nx1) of labels {0,1}, calculated for observations x
    using model parameters w and classification threshold theta
    '''
    sigma = sigmoid(x @ w)
    pred = (sigma >= theta).astype(int).reshape(x.shape[0], w.shape[1])
    return pred


def f_measure(y_true, y_pred):
    '''
    :param y_true: vector of ground truth labels (size: Nx1)
    :param y_pred: vector of predicted labels (size: Nx1)
    :return: value of F-measure
    '''
    TP = np.sum(np.logical_and(y_true, y_pred))
    FP = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    FN = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    f = (2 * TP) / (2 * TP + FP + FN)
    return f


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):

    '''
    :param x_train: trainig set - features (size: NxM)
    :param y_train: training set - labels (size: Nx1)
    :param x_val: validation set - features (size: Nval x M)
    :param y_val: validation set - labels (size: Nval x 1)
    :param w0: initial value of w
    :param epochs: number of SGD iterations
    :param eta: learning rate
    :param mini_batch: mini-batch size
    :param lambdas: list of lambda values that are evaluated in model selection procedure
    :param thetas: list of theta values that are evaluated in model selection procedure
    :return: Functions makes a model selection. It returs tuple (regularization_lambda, theta, w, F), where regularization_lambda
    is the best value of regularization parameter, theta is the best classification threshold, and w is the best model parameter vector.
    Additionally function returns matrix F, which stores F-measures calculated for each pair (lambda, theta).
    REMARK! Use SGD and training criterium with l2 regularization for training.
    '''

    theta = 0
    reg_lambda = 0
    v = 0
    Fmax = - np.inf
    F = np.zeros((len(lambdas), len(thetas)))
    for i, j in enumerate(lambdas):
        obj_fun = lambda w, x, y: regularized_logistic_cost_function(w, x, y, j)
        w, func_values = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        for k, l in enumerate(thetas):
            f = f_measure(y_val, prediction(x_val, w, l))
            F[i, k] = f
            if f > Fmax:
                theta = j
                reg_lambda = l
                v = w
                Fmax = f
    Fmax = F
    return theta, reg_lambda, v, Fmax
