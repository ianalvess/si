from unittest import TestCase
import numpy as np
import os
from si.io.csv_file import read_csv
from datasets import DATASETS_PATH
from si.model_selection.split import train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.ensemble.stacking_classifier import StackingClassifier

class TestStackingClassifier(TestCase):
    @classmethod
    def setUpClass(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', "breast-bin.csv")

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def setUp(self):
        """Configuração por teste: criar os modelos"""
        self.knn1 = KNNClassifier(k=3)
        self.logistic = LogisticRegression()
        self.decision_tree = DecisionTreeClassifier()
        self.knn_final = KNNClassifier(k=5)

        self.stacking_model = StackingClassifier(
            models=[self.knn1, self.logistic, self.decision_tree],
            final_model=self.knn_final
        )

    def test_stacking_classifier_training(self):
        """Teste: verificar se o modelo treina corretamente"""
        self.stacking_model._fit(self.train_dataset)
        self.assertTrue(hasattr(self.stacking_model, '_models'))
        self.assertTrue(hasattr(self.stacking_model.final_model, '_fit'))

    def test_stacking_classifier_predictions(self):
        """Teste: verificar se o modelo faz predições corretamente"""
        self.stacking_model._fit(self.train_dataset)
        X_test, _ = self.test_dataset.X, self.test_dataset.y

        predictions = self.stacking_model._predict(X_test)
        self.assertEqual(len(predictions), len(X_test))  # O número de predições deve ser igual ao número de instâncias

    def test_stacking_classifier_score(self):
        """Teste: calcular a acurácia do modelo no conjunto de teste"""
        self.stacking_model._fit(self.train_dataset)
        X_test, y_test = self.test_dataset.X, self.test_dataset.y

        score = self.stacking_model._score(X_test, y_test)
        self.assertGreaterEqual(score, 0.0)  # A acurácia deve ser >= 0
        self.assertLessEqual(score, 1.0)     # A acurácia deve ser <= 1

