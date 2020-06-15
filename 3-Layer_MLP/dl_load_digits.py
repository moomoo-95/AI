import os
import struct
import numpy as np
# from scipy.special import expit
import sys
import matplotlib.pyplot as plt

# MNIST의 학습 데이터와 테스트 데이터 읽어서 각각 numpy 배열 images, labels로 저장해 리턴하는 함수
def load_mnist(path, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' %kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' %kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>ll', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>llll', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels

# MNIST 데이터 0~9 1개씩 출력
def show_all_digits():
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(10):
        img = X_train[y_train==i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

# MNIST 데이터 숫자 n 25개 출력
def show_n_digits(n):
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(25):
        img = X_train[y_train==n][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

class NeuralMetMLP():
    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, 
    epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0, shuffle=True,
    minibatches=1, random_state=None):
    np.random

# MNIST 학습과 테스트 데이터 읽어오기
X_train, y_train = load_mnist('./MNIST', kind='train')
print('학습 샘플수\t:%d, 컬럼수: %d' %(X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist('./MNIST', kind='t10k')
print('테스트 샘플수\t:%d, 컬럼수: %d' %(X_test.shape[0], X_test.shape[1]))

show_all_digits()
show_n_digits(6)