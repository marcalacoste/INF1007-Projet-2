import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

class JeuDeDonnees:
    colonnes_utiles = [
        "Created Date",
        "Closed Date",
        "Agency",
        "Problem (formerly Complaint Type)",
        "Borough",
        "Latitude",
        "Longitude"
    ]

    def __init__(self, path):
        self.path = path
        self.file = open(path, "r", encoding = "utf8")
        self.data = pd.read_csv(self.file)

    def head(self, quantite_a_garder):
        self.data = self.data.head(quantite_a_garder)

    def show_statistics(self):
        print(self.data.head())
        print(self.data.info())

    def drop_colonnes_inutiles(self):
        for colonne in self.data.columns:
            if colonne not in JeuDeDonnees.colonnes_utiles:
                self.data = self.data.drop(colonne, axis = 1)

    def drop_lignes_incompletes(self):
        self.data = self.data.dropna()

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
    



