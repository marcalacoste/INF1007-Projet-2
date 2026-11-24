from donnees import JeuDeDonnees
from modele import Modele
from interface import preparer_interface_graphique
import os

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FICHIER_DONNEES = os.path.join(RACINE, "donnees", "donnees.csv")
DOSSIER_MODELES = os.path.join(RACINE, "modeles")
FICHIER_META = os.path.join(DOSSIER_MODELES, "metadonnees.json")

donnees = JeuDeDonnees(FICHIER_DONNEES)
donnees.setup_donnees(50000)
donnees.show_statistics()
donnees.creer_graphique()
donnees.show_statistics()

modele = Modele()
modele.forest()
modele.sauvegarder_modele()
modele.charger_modele()
modele.creer_metadonnees(FICHIER_META)

preparer_interface_graphique()