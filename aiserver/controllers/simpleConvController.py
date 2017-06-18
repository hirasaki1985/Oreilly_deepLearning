import sys, os
#sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import json, codecs
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from neuralNetwork.neuralNetwork import NeuralNetwork
from neuralNetwork.Backpropagation import BackPropagation
from neuralNetwork.simple_convnet import SimpleConvNet
from neuralNetwork.ori08_deep_convnet import DeepConvNet

from modules.pklController import PKL
from modules.mnistImageManager import MnistImageManager

class SimpleConvController:
    #pickleFile = "./pklfiles/mymnist_weak.pkl"

    # 初期化
    def __init__(self,
            pickleFile = "./pklfiles/simpleconv_params.pkl"):
        # NeuralNetwork
        # BackPropagation
        """
        self.network = SimpleConvNet(input_dim=(1,28,28), 
            conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
            hidden_size=100, output_size=10, weight_init_std=0.01,
            file_name=pickleFile)
        """
        self.network = DeepConvNet()
        self.network.load_params("./pklfiles/deep_convnet_params.pkl")

        # init
        #print("### SimpleConvController init")

        # setup
        self.imageManager = MnistImageManager()
        
    def predict(self, x):
        img = np.zeros((1, 1, 28, 28))
        #print(img.shape)
        #print(x.shape)
        img[0] = x.reshape(28, 28)
        print(img[0])
        return self.network.predict(img)

    def accuracy(self, x, type = 'mnist'):
        if type == "png":
            print("type is png")
            x = self.imageManager.getMnistImage(x)

        y = self.predict(x)
        print(y)
        return np.argmax(y)

    # from webserver
    def webLogic(self, form):
        print("### Controller webLogic")

        # get png image
        image = Image.open(BytesIO(base64.b64decode(form['file'].value)))

        # logic
        x = self.imageManager.getMnistImage(image)
        y = self.predict(x)
        y = y[0]
        b = y.tolist
        accuracy = json.dumps(list(y))
        max = int(np.argmax(y))

        # result
        result = {
            "y":accuracy, 
            "max": max
        }

        # close
        image.close()

        return json.dumps(result)

