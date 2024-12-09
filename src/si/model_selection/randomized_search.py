import numpy as np
from typing import Dict, Callable, Union, List
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def random_search_cv(model, dataset: Dataset, param_grid: Dict[str, List], scoring: Callable = None, cv: int = 3, n_iter: int = 10) -> Dict[str, Union[float, Dict]]:
    """
    Busca aleatória de hiperparâmetros com validação cruzada.

    Parâmetros:
    ----------
    model : object
        Modelo a ser ajustado.
    dataset : Dataset
        Conjunto de dados para validação.
    param_grid : dict
        Dicionário com os nomes dos hiperparâmetros e seus valores possíveis.
    scoring : callable, opcional
        Função para calcular a pontuação do modelo.
    cv : int
        Número de divisões (folds) na validação cruzada.
    n_iter : int
        Número de combinações aleatórias de hiperparâmetros a testar.

    Retorna:
    --------
    Dict[str, Union[float, Dict]]:
        Dicionário com os melhores hiperparâmetros e a melhor pontuação.
    """
    results = {
        "best_score": -np.inf,
        "best_params": None
    }

    for _ in range(n_iter):
        params = {key: np.random.choice(values) for key, values in param_grid.items()}

        for param, value in params.items():
            setattr(model, param, value)

        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        mean_score = np.mean(scores)

        if mean_score > results["best_score"]:
            results["best_score"] = mean_score
            results["best_params"] = params

    return results

