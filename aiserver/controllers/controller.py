import sys, os
#sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import json, codecs
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from neuralNetwork.neuralNetwork import NeuralNetwork
from neuralNetwork.Backpropagation import BackPropagation


from modules.pklController import PKL
from modules.mnistImageManager import MnistImageManager

class Controller:
    #pickleFile = "./pklfiles/mymnist_weak.pkl"

    # 初期化
    def __init__(self,
            pickleFile = "./pklfiles/mymnist_01.pkl",
            network = NeuralNetwork()):
        # NeuralNetwork
        # BackPropagation

        # init
        print("### Controller init")
        self.pickleFile = pickleFile
        self.network = network

        # setup
        pkl = PKL()
        self.network.setParams(pkl.loadPKL(self.pickleFile))
        self.imageManager = MnistImageManager()
        
    def setParams(self, params):
        self.network.setParams(params)

    def predict(self, x):
        return self.network.predict(x)

    def accuracy(self, x, type = 'mnist'):
        if type == "png":
            print("type is png")
            x = self.imageManager.getMnistImage(x)

        y = self.predict(x)
        return np.argmax(y)

    # from webserver
    def webLogic(self, form):
        print("### Controller webLogic")

        # get png image
        image = Image.open(BytesIO(base64.b64decode(form['file'].value)))

        # logic
        x = self.imageManager.getMnistImage(image)
        y = self.predict(x)
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

