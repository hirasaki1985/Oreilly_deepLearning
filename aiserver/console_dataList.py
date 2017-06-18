import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from controller import Controller
from modules.mnistImageManager import MnistImageManager

# instance
controller = Controller()
imageManager = MnistImageManager()

# imageset load
images, answer = imageManager.getDataList('./images/*.png')
#answer = [2,8,4,1,9,3,5,4]

if images is None:
    print("images is None")

else:
    print("### accuracy ###")
    print ("length")
    print(len(answer))
    print(len(images))

    # accuracy
    print ("accuracy")
    accuracy_cnt = 0
    result = np.zeros((10, 10))
    for i in range(len(images)):
        y = controller.predict(images[i])
        p = np.argmax(y)
        a = answer[i]

        result[p][a] += 1
        print("\nnumber = " + str(i))
        print("softmax")
        print(y)
        print("p = " + str(p))
        print("a = " + str(a))
        if p == a:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(images)))
    print(result)
