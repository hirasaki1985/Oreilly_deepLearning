# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from twoLayers_04 import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#print(x_train.shape) #(60000, 784)
#print(t_train.shape) #(60000, 10)
#print(x_test.shape)  #(10000, 784)
#print(t_test.shape)  #(10000, 10)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# ハイパーパラメータ
iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
result = np.zeros((10, 10))

# １エポックあたりの繰り返し数「100回＝１エポック」
iter_per_epoch = max(train_size / batch_size, 1)

print('train_size = ' + str(train_size))
print('iter_per_epoch = ' + str(iter_per_epoch))
print("### start ###")

for i in range(iters_num):
    # ランダムで選ぶ 60000から100選ぶ
    batch_mask = np.random.choice(train_size, batch_size)
    #print(batch_mask)
    #print(batch_mask.shape) #(100,)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #print(x_batch.shape) #(100, 784)
    #print(t_batch.shape) #(100, 10)
    
    # 勾配の計算np.argmax(t_test[i]
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    #print('### network.numerical_gradient end')
    #print(grad.shape)

    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 損失関数の値を求める
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        #print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


trycount = 1000
accuracy_cnt = 0
for i in range(len(x_test)):
    y = network.predict(x_test[i])
    p = np.argmax(y)
    a = np.argmax(t_test[i])

    #print("p = " + str(p))
    #print("a = " + str(a))
    result[p][a] += 1
    #print(t_test[i])
    if p == a:
        accuracy_cnt += 1

    if (i == trycount):
        break
print("Accuracy:" + str(float(accuracy_cnt) / trycount))
print(result)

# グラフの描画
"""
x = np.arange(len(train_acc_list))
plt.plot(x,train_loss_list)
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.show()
"""
"""
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
"""