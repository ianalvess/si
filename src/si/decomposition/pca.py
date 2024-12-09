from si.base.transformer import Transformer
import numpy as np
class PCA(Transformer):
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean= None
        self.components = None
        self.explained_variance = None

    def _fit(self, X):

        X_centered = X - X.mean(axis=0) # Antes de fazer a covariancia, centraliza-se os dados para facilitar os calculos das relacoes entre elas

        covariance_matrix = np.cov(X_centered, rowvar= False)  # Formula da matriz de covariancia
        eigenvalues, eigenvectores = np.linalg.eig(covariance_matrix) # Calculo dos Autovalores (descobrir variaçao nos dados) e Autovetores (para descobrir o PCA)

        sorted_indices = np.argsort(eigenvalues)[::-1] # Para determinar os componentes que possuem maior variancia temos que ordena-los (decrescente).
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectores = eigenvectores[:,sorted_indices]

        components = eigenvectores[:,:self.n_components] # Selecionamos os componentes principais e suas variancias
        explained_variance = eigenvalues[:self.n_components]

        return components, explained_variance

    def _transform(self, X):

        X_centered = X - self.mean

        X_reduced = np.dot(X_centered, self.components) # transformaçao dos dados, fazendo a multiplicacao das matrizes

        return X_reduced




