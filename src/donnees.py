import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

class JeuDeDonnees:
    """Classe qui contient le jeu de données pour entraîner le modèle"""
    colonnes_utiles = [
        "Created Date",
        "Closed Date",
        "Agency",
        "Problem (formerly Complaint Type)",
        "Borough",
        "Latitude",
        "Longitude",
        "Resolution Time (hours)",
        "Created Month",
        "Created Day",
        "Created Hour",
        "Created Minute"
    ]

    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "r", encoding = "utf8")
        self.data = pd.read_csv(self.file, low_memory = False)

    def __len__(self):
        return len(self.data.index)

    def head(self, quantite_a_garder: int):
        """Sélectionne seulement les premières lignes des données"""
        self.data = self.data.head(quantite_a_garder)

    def show_statistics(self):
        """Affiche dans le terminal différentes satistiques sur le jeu de données"""
        print(self.data.head())
        print(self.data.info())

    def drop_colonnes_inutiles(self):
        """Enlève toutes les colonnes qui ne sont pas dans
        JeuDeDonnes.colonnes_utliles du jeu de données"""
        for colonne in self.data.columns:
            if colonne not in JeuDeDonnees.colonnes_utiles:
                self.data.drop(colonne, axis = 1, inplace = True)

    def drop_lignes_incompletes(self):
        """Enlève toutes les lignes qui manquent de l'information du jeu de données"""
        self.data.dropna(inplace = True)

    def creer_graphique(self):
        """Crée et sauvegarde le graphique d'exploration"""
        data = self.data.groupby("Problem (formerly Complaint Type)")["Created Date"].count()
        labels = []
        for i in range(len(data)):
            if data[data.index[i]] > 500:
                labels.append(data.index[i])
            else:
                labels.append("")
        fig, ax = plt.subplots()
        pie = ax.pie(data.values)
        ax.pie_label(pie, labels = labels, rotate = True, distance = 1.001)
        plt.savefig(os.path.join("donnees", "graphique.png"), bbox_inches = "tight")

    def calculer_temps_resolution(self):
        """Calcule et ajoute le temps de résolution de la plainte au jeu de données"""
        format = "%m/%d/%Y %I:%M:%S %p"
        valeurs = []
        for index, row in self.data.iterrows():
            cree = datetime.datetime.strptime(row["Created Date"], format)
            ferme = datetime.datetime.strptime(row["Closed Date"], format)

            difference = ferme - cree
            difference = difference.total_seconds()/3600
            valeurs.append(difference)

        self.data["Resolution Time (hours)"] = valeurs

        # On enlève les rangées avec un temps négatif
        rows = self.data[self.data["Resolution Time (hours)"] < 0].index
        self.data.drop(rows, inplace = True)

    def ajouter_donnees_creation(self):
        """Ajoute des nouvelles colonnes au jeu de données qui représentent les 
        différentes valeurs du temps de création de la plainte, en int"""
        format = "%m/%d/%Y %I:%M:%S %p"
        mois, jour, heure, minute = [], [], [], []
        for index, row in self.data.iterrows():
            cree = datetime.datetime.strptime(row["Created Date"], format)
            mois.append(cree.month)
            jour.append(cree.day)
            heure.append(cree.hour)
            minute.append(cree.minute)
        
        self.data["Created Month"] = mois
        self.data["Created Day"] = jour
        self.data["Created Hour"] = heure
        self.data["Created Minute"] = minute
        

    def moyenne(self) -> int:
        """Retourne la moyenne du temps de résolution"""
        total = 0
        for index, row in self.data.iterrows():
            total += row["Resolution Time (hours)"]
        return total/len(self)

    def setup_donnees(self, quantite: int):
        """Exécute toutes les étapes pour créer un jeu de données
        qui peut entraîner le modèle ensuite.

        On prend quantite de données dans le jeu de données"""
        self.head(quantite)
        self.drop_colonnes_inutiles()
        self.drop_lignes_incompletes()
        self.calculer_temps_resolution()
        self.ajouter_donnees_creation()

    def enlever_temps_resolution_hauts(self, limite):
        # Test si limite < 0
        # Test si la colonne temps de resolution existe
        rows = self.data[self.data["Resolution Time (hours)"] > limite].index
        self.data.drop(rows, inplace = True)

    



