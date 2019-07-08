# --------------------------------------------------------------------------
# ---  Systems analysis and decision support methods in Computer Science ---
# --------------------------------------------------------------------------
#  Assignment 4: The Final Assignment
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2019
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np


def predict(x):
    """
        Function takes images as the argument. They are stored in the matrix X (NxD). 
        Function returns a vector y (Nx1), where each element of the vector is a class numer {0, ..., 9} associated with recognized type of cloth. 
        :param x: matrix NxD
        :return: vector Nx1
    """
    return np.zeros(shape=(x.shape[0],))
