# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division

import numpy as np
import datetime
import scipy.spatial.distance as dist


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    x = X.toarray().astype(int)
    x_trainT = np.transpose(X_train.toarray()).astype(int)
    return x.shape[1] - x @ x_trainT - (1 - x) @ (1 - x_trainT)


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    order = Dist.argsort(kind='mergesort')
    return y[order]
    # return np.array([y[index_array[int(x / N2), x % N2]] for x in range(N1 * N2)]).reshape((N1, N2))


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """

    number_of_classes = 4
    resized = np.delete(y+1, range(k, y.shape[1]), axis=1)
    summed_with_zero = np.vstack(np.apply_along_axis(
        np.bincount, axis=1, arr=resized, minlength=number_of_classes+1))
    summed = np.delete(summed_with_zero, 0, axis=1)
    return summed / k


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    N1 = np.shape(p_y_x)[0]
    result = 0
    for i in range(N1):
        a = p_y_x[i].tolist()
        if (3 - a[::-1].index(max(a)) != y_true[i]):
            result += 1
    return result / N1


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    sorted_labels = sort_train_labels_knn(
        hamming_distance(Xval, Xtrain), ytrain)
    errors = list(map(lambda k: classification_error(
        p_y_x_knn(sorted_labels, k), yval), k_values))
    min_index = np.argmin(errors)
    return min(errors), k_values[min_index], errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    Ninverse = 1 / ytrain.shape[0]
    array = np.zeros(shape=(4))
    for y in ytrain:
        array[y] += Ninverse
    return array


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    Xtrain = Xtrain.toarray()
    up_factor = a - 1.0
    down_factor = a + b - 2.0

    def f(k, d):
        I_yn_k = ((ytrain+1) == k + 1).astype(bool)
        I_xnd_1 = (Xtrain[:, d] == 1).astype(bool)
        up = up_factor + np.count_nonzero(I_yn_k & I_xnd_1)
        down = down_factor + np.count_nonzero(I_yn_k)
        return up / down

    g = np.vectorize(f)
    return np.fromfunction(g, shape=(4, Xtrain.shape[1]), dtype=int)
 # def f(k, d):
    #     up = upAddition + sum((ytrain == k + 1) & (Xtrain[:, d] == 1))
    #     down = downAddition + a_priori[k]
    #     # for n in range(N):
    #     #     if ((ytrain[n] == k + 1) and (Xtrain[n, d] == 1)):
    #     #         up += 1.0
    #
    #     return up / down
    #
    # g = np.vectorize(f)
    # return np.fromfunction(g, shape=(4, Xtrain.shape[1]), dtype=int)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X = X.toarray()
    p_x_1_y_rev = 1 - p_x_1_y
    X_rev = 1 - X
    res = []
    for i in range(X.shape[0]):
        success = p_x_1_y ** X[i, ]
        fail = p_x_1_y_rev ** X_rev[i, ]
        a = np.prod(success * fail, axis=1) * p_y
        # suma p(x|y') * p(y')
        sum = np.sum(a)
        # prawdopodobieÅ„stwo kaÅ¼dej z klas podzielone przez sumÄ™
        res.append(a / sum)
    return np.array(res)


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    A = len(a_values)
    B = len(b_values)

    p_y = estimate_a_priori_nb(ytrain)

    def f(a, b):
        p_x_y_nb = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
        p_y_x = p_y_x_nb(p_y, p_x_y_nb, Xval)
        err = classification_error(p_y_x, yval)
        return err

    g = np.vectorize(f)
    errors = np.fromfunction(g, shape=(A, B), dtype=int)

    min = np.argmin(errors)
    minA = min // A
    minB = min % A
    return (errors[minA, minB], a_values[minA], b_values[minB], errors)
