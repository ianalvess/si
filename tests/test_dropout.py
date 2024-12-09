import numpy as np
from unittest import TestCase
from si.neural_networks.layers import Dropout

class TestDropoutLayer(TestCase):
    def setUp(self):
        """Configurações antes de cada teste."""
        self.probability = 0.5
        self.input_data = np.random.rand(4, 4)
        self.dropout_layer = Dropout(self.probability)

    def test_training_mode(self):
        """Teste do comportamento em modo de treinamento."""
        output = self.dropout_layer.forward_propagation(self.input_data, training=True)

        self.assertTrue(np.any(output == 0), "Nenhum neurônio foi desativado no modo de treinamento.")

        self.assertTrue(np.all((output == 0) | (output == self.input_data)),
                        "Os neurônios ativos não correspondem ao input no modo de treinamento.")

    def test_inference_mode(self):
        """Teste do comportamento em modo de inferência."""
        output = self.dropout_layer.forward_propagation(self.input_data, training=False)

        np.testing.assert_array_almost_equal(output, self.input_data,
                                             err_msg="A saída no modo de inferência foi alterada.")

    def test_mask_generation(self):
        """Teste da geração da máscara binomial no modo de treinamento."""
        self.dropout_layer.forward_propagation(self.input_data, training=True)
        mask = self.dropout_layer.mask

        self.assertTrue(np.all((mask == 0) | (mask == 1)), "A máscara contém valores diferentes de 0 ou 1.")


        active_rate = np.mean(mask)
        expected_active_rate = 1 - self.probability
        self.assertAlmostEqual(active_rate, expected_active_rate, delta=0.1,
                               msg="A taxa de ativação da máscara não corresponde à esperada")