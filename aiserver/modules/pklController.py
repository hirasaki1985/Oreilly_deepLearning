import sys, os
import pickle
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from neuralNetwork.neuralNetwork import NeuralNetwork

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメータ
#iters_num = 10000  # 繰り返しの回数を適宜設定する
#train_size = x_train.shape[0]
#batch_size = 100
#learning_rate = 0.1

# １エポックあたりの繰り返し数「100回＝１エポック」

class PKL:
    def __init__(self):
        print("### PKL init ###")
        self.params = {};

    def getRandomPKL(self, iters_num = 10000, train_size = 60000, batch_size = 100, learning_rate = 0.1, input_size=784, hidden_size=50, output_size=10):
        print("### PKL getRandomPKL ###")
        # init
        self.network = NeuralNetwork()
        self.setNetworkParams(iters_num, train_size, batch_size, learning_rate)
        self.setRandomPKL(input_size, hidden_size, output_size)

        # calc
        self.network.setParams(self.params)
        self.execute()

        return self.getParams()

    def setNetworkParams(self, iters_num, train_size, batch_size, learning_rate):
        self.iters_num = iters_num
        self.train_size = train_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iter_per_epoch = max(train_size / batch_size, 1)
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        print("iters_num = " + str(self.iters_num))
        print("train_size = " + str(self.train_size))
        print("batch_size = " + str(self.batch_size))
        print("learning_rate = " + str(self.learning_rate))
        print("iter_per_epoch = " + str(self.iter_per_epoch))

    def loadPKL(self, filename):
        print("### PKL loadPKL ###")
        print(filename)
        data = {}
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        return data

    def savePKL(self, filename, data):
        print("### PKL loadPKL ###")
        print(filename)
        with open(filename,'wb') as f1:
            pickle.dump(data, f1)	#pickle.dump（データ、ファイル）

    def getParams(self):
        return self.network.params

    def setRandomPKL(self, input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01):
        print("### PKL setRandomPKL ###")
        print ("input_size = " + str(input_size))
        print ("hidden_size = " + str(hidden_size))
        print ("output_size = " + str(output_size))
        print ("weight_init_std = " + str(weight_init_std))

        # 重みの初期化
        # [784, 50]
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def execute(self):
        print("### PKL execute ###")
        for i in range(self.iters_num):
            # ランダムで選ぶ 60000から100選ぶ
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            #print(batch_mask)
            #print(batch_mask.shape) #(100,)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            #print(x_batch.shape) #(100, 784)
            #print(t_batch.shape) #(100, 10)

            # 勾配の計算np.argmax(t_test[i]
            #grad = network.numerical_gradient(x_batch, t_batch)
            grad = self.network.gradient(x_batch, t_batch)
            #print('### network.numerical_gradient end')
            #print(grad.shape)

            # パラメータの更新
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.network.params[key] -= self.learning_rate * grad[key]

                # 損失関数の値を求める
                loss = self.network.loss(x_batch, t_batch)
                self.train_loss_list.append(loss)

            # 1エポックごとに認識精度を計算
            if i % self.iter_per_epoch == 0:
                train_acc = self.network.accuracy(x_train, t_train)
                test_acc = self.network.accuracy(x_test, t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                #print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
