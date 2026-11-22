import sys
sys.path.insert(0, "src")
from donnees import JeuDeDonnees
import pytest
import os

donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))


def test_dataset_reduit():
    donnees.head(50000)
    assert len(donnees) <= 50000

def test_colonnes_inutiles():
    donnees.setup_donnees(50000)
    assert len(donnees.data.columns) == len(JeuDeDonnees.colonnes_utiles)

def test_lignes_incompletes():
    donnees.head(50000)
    donnees.drop_lignes_incompletes()
    assert donnees.data.isnull().any().any() == False

def test_graphique():
    donnees.head(50000)
    donnees.drop_colonnes_inutiles()
    donnees.drop_lignes_incompletes()
    donnees.creer_graphique()
    assert os.path.exists(os.path.join("donnees", "graphique.png")) == True

def test_limite_temps_resolution():
    donnees.setup_donnees(50000)
    with pytest.raises(ValueError):
        donnees.enlever_temps_resolution_hauts(-1)
        donnees.enlever_temps_resolution_hauts(0)

def test_colonne_temps_resolution_hauts():
    with pytest.raises(ValueError):
        donnees.enlever_temps_resolution_hauts(1000)