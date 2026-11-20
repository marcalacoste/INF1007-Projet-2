from donnees import JeuDeDonnees
from modele import Modele
import os

donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
donnees.setup_donnees(50000)
donnees.show_statistics()
donnees.creer_graphique()

modele = Modele()
modele.forest()
modele.sauvegarder_modele()
modele.charger_modele()
modele.creer_metadonnees(os.path.join("modeles", "metadonnees.json"))