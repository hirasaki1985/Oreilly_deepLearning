# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient
from neuralNetwork.neuralNetwork import NeuralNetwork

class BackPropagation(NeuralNetwork):
    # 初期化を行う
    #def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #print(self.params)
    def __init__(self):
        print("BackPropagation init()")

    # 重み、バイアスパラメータをセット
    def setParams(self, params):
        # 重みの初期化
        # [784, 50]
        self.params = params

    def setLayers():
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()   

    # 認識(推論)を行う。xは画像データ
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # 一層目の計算
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        # 二層目の計算
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 損失関数の値を求める
    # x:入力データ(画像データ), t:教師データ(正解ラベル)
    def loss(self, x, t):
        #️ y = ソフトマックス関数の結果
        y = self.predict(x)

        #　交差エントロピー誤差の結果を返す
        return cross_entropy_error(y, t)

    # 認識精度を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    #️⃣ 重みパラメータに対する勾配を求める
    # x:入力データ(画像データ), t:教師データ
    def numerical_gradient(self, x, t):
        # 損失関数の結果
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 重みパラメータに対する勾配を求める
    # numerical_gradient()の高速版！
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
