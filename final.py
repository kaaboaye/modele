import pickle
import numpy as np

with open('A4_LAB_EN/train.pkl', 'rb') as f:
    (train, labels) = pickle.load(f)


train_images, test_images = (train[:50000, :], train[50000:, :])
train_labels, test_labels = (labels[:50000], labels[50000:])

train_images = train_images.reshape(50000, 36, 36, 1)
test_images = test_images.reshape(10000, 36, 36, 1)

pkl_file = open('weights_1_32_final.pkl', 'rb')
all_weights = pickle.load(pkl_file)
pkl_file.close()


def relu(x):
    return np.maximum(0.0, x)


def conv(inputs, strides, shape, filters):
    size = int((inputs.shape[1] - shape[0]) / strides + 1)
    features_map = np.zeros((inputs.shape[0], size, size, shape[-1]))
    for x in range(size):
        for y in range(size):
            features_map[:, x, y, :] = np.sum(inputs[
                :,
                x*strides: x*strides + shape[0],
                y*strides: y*strides + shape[1], :, np.newaxis
            ] * filters,
                axis=(1, 2, 3)
            )
    return relu(features_map)


def flatten(inputs):
    return np.reshape(inputs, (inputs.shape[0], -1))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dense(inputs, weights):
    return sigmoid(inputs.dot(weights))


def predict_one(inp):
    res = conv(np.reshape(inp, (1, 36, 36, 1)),
               1, (3, 3, 1, 32), all_weights[0])
    # res = conv(res, 1, (3, 3, 16, 16), all_weights[1])
    res = flatten(res)
    res = dense(res, all_weights[1])
    return np.argmax(res)


def predict(x):
    """
        Function takes images as the argument. They are stored in the matrix X (NxD). 
        Function returns a vector y (Nx1), where each element of the vector is a class numer {0, ..., 9} associated with recognized type of cloth. 
        :param x: matrix NxD
        :return: vector Nx1
    """
    x = x.reshape(x.shape[0], 36, 36, 1)
    return np.reshape(np.array([predict_one(x[i, :, :, :]) for i in range(x.shape[0])]), (-1, 1))


res = predict(test_images[:2500])
print(res)
