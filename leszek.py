import pickle
import numpy as np

with open('A4_LAB_EN/train.pkl', 'rb') as f:
    (train, labels) = pickle.load(f)

train_images, test_images = (train[:50000, :], train[50000:, :])
train_labels, test_labels = (labels[:50000], labels[50000:])

train_images = train_images.reshape(50000, 36, 36, 1)
test_images = test_images.reshape(10000, 36, 36, 1)


def predict(x):
    output = list()
    for i in range(x.shape[0]):
        output.append(1)
    output = np.array(output)
    output = np.asarray(output)
    output = np.reshape(output, (-1, 1))
    return output


res = predict(test_images[:100])
print(res)
