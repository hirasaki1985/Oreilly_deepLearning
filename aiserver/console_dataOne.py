import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from controllers.controller import Controller
from controllers.simpleConvController import SimpleConvController
from modules.mnistImageManager import MnistImageManager

# instance
#controller = Controller()
controller = SimpleConvController()
imageManager = MnistImageManager()

# imageset load
image = imageManager.getMnistDataFromPng('./images/001_2.png')
answer = 2

#print(type(image))
if image is None:
    print("image is None")
else:
    print(image)
    # accuracy
    print("accuracy")
    p = controller.accuracy(image)
    a = answer
    print(p)
    print(a)
    if p == a:
        print("")
