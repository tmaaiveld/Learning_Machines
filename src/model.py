"""
Placeholder class for a Keras model object. Returns random numbers.
"""

import random


class Model:
    def predict(self, S):
        return [random.random() for i in range(4)]

    def train(self, X, y):
        pass
