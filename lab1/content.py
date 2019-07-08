# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Linear regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial as pol


def mean_squared_error(input, output, model):
    '''
    :param x: input vector Nx1
    :param y: output vector Nx1
    :param w: model parameters (M+1)x1
    :return: mean squared error between output y
    and model prediction for input x and parameters w
    '''

    training_matrix = design_matrix(input, model.shape[0] - 1)

    errors_sum = 0
    for i in range(output.shape[0]):
        errors_sum += (output[i] - training_matrix[i] @ model) ** 2

    error = errors_sum / input.shape[0]
    return np.array(*error)


def design_matrix(x_train, M):
    '''
    :param x_train: input vector Nx1
    :param M: polynomial degree 0,1,2,...
    :return: Design Matrix Nx(M+1) for M degree polynomial
    '''
    result = np.ndarray((x_train.size, M + 1))

    for row in range(x_train.size):
        for degree in range(M + 1):
            result[row][degree] = x_train[row] ** degree

    return result


def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''

    traning_matrix = design_matrix(x_train, M)
    traning_matrix_transposed = traning_matrix.transpose()

    model = np.linalg.inv(traning_matrix_transposed @
                          traning_matrix) @ traning_matrix_transposed @ y_train

    error = mean_squared_error(x_train, y_train, model)
    return (model, error)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''
    training_matrix = design_matrix(x_train, M)
    training_matrix_transposed = training_matrix.transpose()

    model = np.linalg.inv(training_matrix_transposed @ training_matrix + (
        regularization_lambda * np.identity(M + 1))) @ training_matrix_transposed @ y_train

    error = mean_squared_error(x_train, y_train, model)
    return (model, error)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M_values: array of polynomial degrees that are going to be tested in model selection procedure
    :return: tuple (w,train_err, val_err) representing model with the lowest validation error
    w: model parameters, train_err, val_err: training and validation mean squared error
    '''
    train_models = []
    train_models_errors = []
    for m in M_values:
        w, err = least_squares(x_train, y_train, m)
        train_models.append(w)
        train_models_errors.append(err)

    errors = []
    for m in M_values:
        errors.append(mean_squared_error(
            x_val, y_val, train_models[m]))

    minimal_training_error = min(errors)
    minimal_error = errors.index(minimal_training_error)

    error = train_models_errors[minimal_error]
    model = train_models[minimal_error]

    return (model, error, minimal_training_error)


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedurei
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''

    train_arr = []
    for lambd in lambda_values:
        w, err = regularized_least_squares(x_train, y_train, M, lambd)
        train_arr.append((w, err, lambd))

    errors = []
    for w in train_arr:
        errors.append(mean_squared_error(x_val, y_val, w[0]))

    minimal_training_error = min(errors)
    minimal_error = errors.index(minimal_training_error)

    w, train_err, lambd = train_arr[minimal_error]
    return (w, train_err, minimal_training_error, lambd)
