from donnees import JeuDeDonnees
import os

donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
donnees.head(50000)
donnees.drop_colonnes_inutiles()
donnees.drop_lignes_incompletes()
donnees.show_statistics()
donnees.creer_graphique()

