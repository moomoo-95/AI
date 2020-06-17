import os
import struct
import numpy as np
# import sys
import matplotlib.pyplot as plt
# import pickle
import time

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

# 다층 퍼셉트론 구현
# n_output(출력층 뉴런 개수), n_features(입력층 뉴런 개수), n_hidden(은닉층 뉴런 개수)
# 오버피팅 방지 : l1(L1 Regularization, 정규화 람다값) l2(L2 Regularization, 정규화 람다값)
# epochs(학습 반복 횟수), eta(학습률), alpha(모멘텀 학습 파라미터)
# 학습률 동적 변화 : decrease_const(학습률을 점점 작아지게 하는 용도로 사용)
# shuffle(매 반복마다 트레이닝 데이터 셔플 여부)
# minibatches(매 학습당 추출 트레이닝 데이터 갯수, 확률적 경사하강법)
class NeuralMetMLP():
    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, 
                epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0,
                shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.cost_ = []
    # one hot encoding, 타겟 값 설정
    def _encode_labels(self, y, k): # y는 실제값, k는 출력층의 노드 개수
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        
        return onehot
    # 입력층과 은닉층 사이 가중치(w1), 은닉층과 출력층 사이 가중치(w2) 초기화 및 편향 포함
    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features+1))
        w1 = w1.reshape(self.n_hidden, self.n_features+1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden+1))
        w2 = w2.reshape(self.n_output, self.n_hidden+1)

        return w1, w2
    # Sigmoid 함수
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    # Sigmoid 함수의 미분값
    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg*(1-sg)
    def _softmax(self, a) : 
        c = np.max(a) # 최댓값
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
    def _relu(self, x):
        mask = (x <= 0)    # x <= 0 : True, x > 0 : False
        out = x.copy()
        out[mask] = 0
        return out
    def _back_relu(self, dout):
        mask = (dout <= 0)
        dout[mask] = 0
        dx = dout
        return dx
    # X에 편향 값 추가
    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1]+1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0]+1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('how must be "column" or "row"')

        return X_new
    # w1, w2 가중치로 X를 순전파 시킴(forward propagation)
    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)

        return a1, z2, a2, z3, a3
    # L2 Regularization 값 리턴
    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0)*(np.sum(w1[:, 1:]**2)+np.sum(w2[:, 1:]**2))
    # L1 Regularization 값 리턴
    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0)*(np.abs(w1[:, 1:]**20).sum()+np.abs(w2[:, 1:]**2).sum())
    #  비용함수 리턴
    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc*(np.log(output))
        term2 = (1 - y_enc)*np.log(1-output)
        cost = np.sum(term1-term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term

        return cost
    # 역전파 알고리즘 (back propagation) 파라미터에 대한 그래디언트 계산
    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        #역전파 알고리즘
        delta3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        delta2 = w2.T.dot(delta3)*self._sigmoid_gradient(z2)
        delta2 = delta2[1:, :]
        grad1 = delta2.dot(a1)
        grad2 = delta3.dot(a2.T)

        #정규화
        grad1[:, 1:] += (w1[:, 1:]*(self.l1+self.l2))
        grad2[:, 1:] += (w2[:, 1:]*(self.l1+self.l2))

        return grad1, grad2
    # X 예측값 리턴
    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)

        return y_pred
    def accurancy_check(self, opt=True):
        y_train_pred = self.predict(X_validation)
        success=np.sum(y_validation == y_train_pred, axis=0)
        total = X_validation.shape[0]
        accurancy = success/total
        if opt:
            print('예측성공/총개수: [%d]/[%d]' %(success, total))
            print('accurancy: %.2f%%' %(accurancy*100))
            return 0
        else:
            return (accurancy*100)
    # 트레이닝 데이터 X, y로 MLP 학습
    def fit(self, X, y, print_progress=False):
        X_data, y_data = X.copy(), y.copy()
        # 타겟값 설정
        y_enc = self._encode_labels(y, self.n_output)

        DELTA_w1_prev = np.zeros(self.w1.shape)
        DELTA_w2_prev = np.zeros(self.w2.shape)
        start = time.time()
        last_epochs = self.epochs
        accurancy = []
        for i in range(self.epochs):

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const*i)
            # epoch 당 평균 에러율
            avg_err = 0.
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3,
                                        w1=self.w1, w2=self.w2)
                avg_err += (cost / self.minibatches)/100

                #역전파를 통해 가중치 업데이트를 위한 미분값 계산
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2,
                                                    y_enc=y_enc[:, idx],
                                                    w1=self.w1, w2=self.w2)
                
                #가중치 업데이트
                DELTA_w1, DELTA_w2 = self.eta*grad1, self.eta*grad2
                self.w1 -= (DELTA_w1 + (self.alpha * DELTA_w1_prev))
                self.w2 -= (DELTA_w2 + (self.alpha * DELTA_w2_prev))
                DELTA_w1_prev, DELTA_w2_prev = DELTA_w1, DELTA_w2
            self.cost_.append(avg_err)
            if print_progress:
                print("Epoch:", '%02d' % (i+1), "error=", "{:.9f}".format(avg_err))
            accurancy.append(self.accurancy_check(False))
            if self.accurancy_check(False) >= finish:
                last_epochs = i
                break
        # plt.scatter(range(1, self.epochs +1), self.cost_)
        print("Training Time :", "{:.2f}".format(time.time() - start), "s")
        plt.subplot(211)
        plt.plot(range(1, last_epochs + 2), self.cost_)
        plt.ylabel('Error Rate(%)')
        plt.xlabel('Epochs')
        plt.subplot(212)
        plt.plot(range(1, last_epochs + 2), accurancy)
        plt.ylabel('Accurancy Rate(%)')
        plt.xlabel('Epochs')
        self.accurancy_check(True)
        return self
    

# MNIST 학습과 테스트 데이터 읽어오기
X_train, y_train = load_mnist('./MNIST', kind='train')
# print('학습 샘플수\t:%d, 컬럼수: %d' %(X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('./MNIST', kind='t10k')
# print('테스트 샘플수\t:%d, 컬럼수: %d' %(X_test.shape[0], X_test.shape[1]))
VALIDATION_SIZE = 5000
X_validation = X_train[:VALIDATION_SIZE, ...]
y_validation = y_train[:VALIDATION_SIZE]
# show_all_digits()
# show_n_digits(6)
# order=['learning_rate=0.0001', 'learning_rate=0.001', 'learning_rate=0.01']
order=['MLP']
finish = 97.0
mlp = NeuralMetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50,
                    l1=0.0, l2=0.1, epochs=1000, eta=0.001, alpha=0.001,
                    decrease_const=0.00001, shuffle=True, minibatches=100, random_state=1)
# mlp2 = NeuralMetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50,
#                     l1=0.0, l2=0.1, epochs=100, eta=0.001, alpha=0.001,
#                     decrease_const=0.00001, shuffle=True, minibatches=100, random_state=1)
# mlp3 = NeuralMetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50,
#                     l1=0.0, l2=0.1, epochs=100, eta=0.01, alpha=0.001,
#                     decrease_const=0.00001, shuffle=True, minibatches=100, random_state=1)

mlp.fit(X_train, y_train, print_progress=True)
# mlp2.fit(X_train, y_train, print_progress=True)
# mlp3.fit(X_train, y_train, print_progress=True)
# with open(os.path.join('./MNIST', 'mlp_digits.pkl'), 'wb') as f:
#     pickle.dump(mlp, f, protocol=4)
# print('학습 데이터 저장 완료')

# with open(os.path.join('./MNIST', 'mlp_digits.pkl'), 'rb') as f:
#     pickle.load(f)
# print('학습 데이터 로드 완료')


# plt.ylabel('Error Rate')
# plt.ylabel('Accurancy Rate(%)')
# plt.xlabel('Epochs')
# plt.legend(order)
plt.show()