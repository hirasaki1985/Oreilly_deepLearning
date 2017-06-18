# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from neuralNetwork.simple_convnet import SimpleConvNet
from common.trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01,
                        file_name="./pklfiles/simpleconv_params.pkl")

# accuracy
trycount = 2000
accuracy_cnt = 0
result = np.zeros((10, 10))
batch_size=1

for i in range(len(x_test)):
    if (i == trycount):
      break

    batch_mask = np.random.choice(x_test.shape[0], batch_size)
    x_test_batch = x_test[batch_mask]
    t_test_batch = t_test[batch_mask]

    p = network.predict(x_test_batch)

    #testprint
    #print("### predict result")
    #print(p.shape)
    #print(p)
    #print(t_test_batch.shape)
    #print(t_test_batch)

    for j in range(len(p)):
      out = np.argmax(p[j])
      ans = t_test_batch[j]

      result[out][ans] += 1
      print("i = " + str(i) + ": j = " + str(j) + ": out = " + str(out) + ": ans = " + str(ans))

      if out == ans:
          accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / trycount))
print("trycount = " + str(trycount))
print("accuracy_cnt = " + str(accuracy_cnt))
print(result)


"""                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()
"""
# パラメータの保存
#network.save_params("params.pkl")
#print("Saved Network Parameters!")

# グラフの描画
