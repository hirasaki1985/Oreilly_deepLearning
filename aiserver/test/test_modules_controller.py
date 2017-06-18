import sys, os
import unittest
sys.path.append('./')
from modules.controller import Controller
from modules.mnistImageManager import MnistImageManager

class TestController(unittest.TestCase):
    def __init__(self):
        self.target = Controller()
        self.imageManager = MnistImageManager()

    def test_logic(self):
        image = self.imageManager.getMnistDataFromPng('./images/001_2.png')
        params = {
            "judgeimage":image
        }
        result = self.target.logic(params)
        self.assertEqual(result.accuracy, 2)

test = TestController()
test.test_logic()
