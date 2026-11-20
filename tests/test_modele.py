"""Tests unitaires des méthodes de référence et d'évaluation du modèle.

Les tests utilisent des données synthétiques et ne chargent pas le CSV NYC 311.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modele import Modele


def modele_sans_chargement(y_train: list[float], y_test: list[float]) -> Modele:
    """Construit une instance minimale sans lire le CSV ni entraîner de forêt."""
    modele = Modele.__new__(Modele)
    modele.X_train = pd.DataFrame({"feature": range(len(y_train))})
    modele.y_train = pd.Series(y_train, dtype=float)
    modele.X_test = pd.DataFrame({"feature": range(len(y_test))})
    modele.y_test = pd.Series(y_test, dtype=float)
    return modele


def test_baseline_utilise_moyenne() -> None:
    """La référence doit prédire la moyenne du jeu d'entraînement."""
    modele = modele_sans_chargement([2, 4, 100], [1, 7])
    assert modele.baseline() == pytest.approx([35+1/3, 35+1/3])


def test_baseline_taille_predictions() -> None:
    """La référence fournit une prédiction pour chaque ligne de test."""
    modele = modele_sans_chargement([1, 3, 5, 7], [2, 8, 10])
    predictions = modele.baseline()
    assert len(predictions) == 3
    assert np.isfinite(predictions).all()


def test_mae_valeur_connue() -> None:
    """La MAE correspond à la moyenne des écarts absolus."""
    modele = modele_sans_chargement([1, 2], [2, 4, 8])
    assert modele.mae(np.array([3, 1, 10])) == pytest.approx(2.0)


def test_mae_predictions_exactes() -> None:
    """Une prédiction parfaite produit une MAE nulle."""
    modele = modele_sans_chargement([1, 2], [2, 4, 8])
    assert modele.mae(np.array([2, 4, 8])) == pytest.approx(0.0)


def test_mae_taille_incompatible() -> None:
    """Une erreur est levée si la taille des prédictions diffère."""
    modele = modele_sans_chargement([1, 2], [2, 4, 8])
    with pytest.raises(ValueError):
        modele.mae(np.array([2, 4]))


def test_mae_valeur_negative() -> None:
    """La métrique prend la valeur absolue des écarts, même sous zéro."""
    modele = modele_sans_chargement([1, 2], [-2, 2])
    assert modele.mae(np.array([2, -2])) == pytest.approx(4.0)
