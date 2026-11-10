import sys
sys.path.insert(0, "src")
from donnees import JeuDeDonnees
import numpy as np
import os


def test_dataset_reduit():
    donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
    donnees.head(50000)
    assert len(donnees) <= 50000

def test_colonnes_inutiles():
    donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
    donnees.setup_donnees(50000)
    assert len(donnees.data.columns) == len(JeuDeDonnees.colonnes_utiles)

def test_lignes_incompletes():
    donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
    donnees.head(50000)
    donnees.drop_lignes_incompletes()
    assert donnees.data.isnull().any().any() == False

def test_graphique():
    donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
    donnees.head(50000)
    donnees.drop_colonnes_inutiles()
    donnees.drop_lignes_incompletes()
    donnees.creer_graphique()
    assert os.path.exists(os.path.join("donnees", "graphique.png")) == True