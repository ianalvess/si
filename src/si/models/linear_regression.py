from si.base.model import Model
from si.data import dataset
from si.data.dataset import Dataset
from si.metrics import mse
import numpy as np

class RidgeRegression(Model):

    def __init__(self, l2_penalty:float, alpha:float, max_iter:int, patience:int, scale:bool, **kwargs):
        super().__init__(**kwargs)

        """
        Parameters
        ----------
        l2_penalty
        alpha
        max_iter
        patience
        scale
        ----------
        """
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = {}

    def fit_(self, dataset:Dataset) -> 'RidgeRegression':

        if self.scale:
            self.mean = dataset.get_mean() #np.mean(dataset.X, axis = 0)
            self.std = np.nanstd(dataset.X, axis = 0)

            x = (dataset.X - self.mean)/self.std

        else:
            x = dataset.X

        m,n = dataset.X.shape()  # m, n numeros de linhas e colunas
        self.theta = np.zeros(n)

        i = 0
        early_stopping = 0
        while i < self.max_iter and early_stopping < self.patience:

            Y_pred = np.dot(self.theta, dataset.X) + self.theta_zero

            gradient = (self.alpha/m) * np.dot((y_pred - dataset.y), x)
            penalty_term_gradient = self.theta * (1 - self.alpha * self.l2_penalty / m)


            self.theta = penalty_term_gradient - gradient
            self.theta_zero =  self.theta_zero - gradient

            Y_pred = np.dot(self.theta, dataset.X) + self.theta_zero # atualizamos o valor de y_pred

            cost = (1/(2*m)) * np.sum((Y_pred - dataset.y)**2) + (self.l2_penalty * np.sum(self.theta ** 2))

            self.cost_history[i] = cost
            if i >0 and self.cost_history[i-1] < cost:
                early_stopping += 1

            else:
                early_stopping = 0

            i += 1

        return self

    


