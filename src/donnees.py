import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

class JeuDeDonnees:
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

    def __init__(self, path):
        self.path = path
        self.file = open(path, "r", encoding = "utf8")
        self.data = pd.read_csv(self.file)

    def __len__(self):
        return len(self.data.index)

    def head(self, quantite_a_garder):
        self.data = self.data.head(quantite_a_garder)

    def show_statistics(self):
        print(self.data.head())
        print(self.data.info())

    def drop_colonnes_inutiles(self):
        for colonne in self.data.columns:
            if colonne not in JeuDeDonnees.colonnes_utiles:
                self.data.drop(colonne, axis = 1, inplace = True)

    def drop_lignes_incompletes(self):
        self.data.dropna(inplace = True)

    def creer_graphique(self):
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
        return

    def ajouter_donnees_creation(self):
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
        

    def moyenne(self):
        total = 0
        for index, row in self.data.iterrows():
            total += row["Resolution Time (hours)"]
        return total/len(self)

    def setup_donnees(self, quantite):
        self.head(quantite)
        self.drop_colonnes_inutiles()
        self.drop_lignes_incompletes()
        self.calculer_temps_resolution()
        self.ajouter_donnees_creation()
    



