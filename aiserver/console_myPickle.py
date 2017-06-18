import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from controllers.controller import Controller

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# instance
controller = Controller()

# accuracy
trycount = 1000
accuracy_cnt = 0
result = np.zeros((10, 10))

for i in range(len(x_test)):
    p = controller.accuracy(x_test[i])
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
