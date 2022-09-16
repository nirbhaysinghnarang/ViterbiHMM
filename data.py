from cgi import test
import os
import json


class Data:
    def __init__(self, trainPath, testPath, valPath):
        self.train = self.readFile(trainPath)
        self.val = self.readFile(valPath)
        self.test = self.readFile(testPath)

    def readFile(self, path):
        with open(os.path.join(os.getcwd(), path), 'r') as f:
            return json.loads(f.read())
