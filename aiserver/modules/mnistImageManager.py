import numpy as np
from PIL import Image
import struct
import glob

class MnistImageManager:
	#def __init__(self):
	def getMnistDataFromPng(self, filename):
		print("### MnistImageManager getMnistDataFromPng")
		print("filename = " + filename)
		#file = glob.glob(filename)
		# make Label Data
		#lbl = file.split('_')
		#lbl = int(lbl[1])
		#labl += struct.pack('B', lbl)

		image = Image.open(filename)
		img = self.getMnistImage(image)

		#d = np.array(img)
		#d = np.delete(d, 3, 2)   # index=3, axis=2 : RGBA -> RGB
		#d = np.mean(d, 2)        # RGB -> L
		#d = np.swapaxes(d, 0, 1)
		#d = np.uint8(d.T.flatten('C'))    # [y][x] -> [y*x]
		#ld = d.tolist()          # numpy object -> python list object
		#append(struct.pack('784B', *ld))
		#print(img.size)
		return img

	def getMnistImage(self, image):
		#image = image.convert('L').resize((28, 28), c)
		#image = image.convert('L').resize(28, 28)
		image = image.convert('L').resize((28, 28), Image.ANTIALIAS)
		#image.show()
		img = np.asarray(image, dtype=float)
		img = np.floor(56 - 56 * (img / 256))
		img = img.flatten()
		#print(img)
		return img

	def getDataList(self, filter):
		print("### MnistImageManager getDataList")
		print("filter = " + filter)
		data = []
		label = []
		files = glob.glob(filter)
		#print(files)
		files.sort()
		for one in files:
			data.append(self.getMnistDataFromPng(one))
			start = one.rfind('_')
			end = one.rfind('.')
			#print (str(start) + " : " + str(end))
			#print (one[start + 1:start - end -2])
			label.append(int(one[start + 1:start - end -2]))
		print(label)
		return data, label
