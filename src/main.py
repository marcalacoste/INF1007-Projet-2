from donnees import JeuDeDonnees
import os

donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
donnees.setup_donnees(50000)
donnees.show_statistics()
donnees.creer_graphique()