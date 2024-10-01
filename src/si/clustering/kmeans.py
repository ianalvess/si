import numpy as np
from si.base.transformer import Transformer

class Kmeans:
    def __init__(self,Transformer,Model,k,max_iter,distance):
        self.Transformer = Transformer
        self.Model = Model
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.centroides = None
        self.labels = None

    def _fit(self, X):
        centroides = np.ramdom.permutation(self.k)







    def _transform(self):
        pass

    def _predict(self):
        pass