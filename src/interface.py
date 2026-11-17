"""Interface du projet INF1007 : estimation du délai de résolution des plaintes 311."""

import csv
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modele import Modele


DOSSIER = "modeles"
FICHIER_MODELE = os.path.join(DOSSIER, "modele_311.joblib")
FICHIER_META = os.path.join(DOSSIER, "metadonnees.json")
FICHIER_HISTORIQUE = os.path.join(DOSSIER, "historique_predictions.csv")
COLONNE_PROBLEME = "Problem (formerly Complaint Type)"


class SaisieInvalideError(Exception):
    """Signale une caractéristique manquante ou invalide."""


class Variable(ABC):
    """Décrit une variable que l'utilisateur ou l'application doit fournir."""

    def __init__(self, nom: str):
        self._nom = nom

    def __str__(self) -> str:
        return self._nom

    @abstractmethod
    def valider(self, valeur):
        """Vérifie une valeur et retourne sa représentation utilisable."""


class VariableCategorielle(Variable):
    """Vérifie qu'une catégorie fait partie des choix disponibles."""

    def __init__(self, nom: str, choix: list[str]):
        super().__init__(nom)
        self._choix = choix

    def valider(self, valeur) -> str:
        if valeur not in self._choix:
            raise SaisieInvalideError(f"{self} : sélectionne une catégorie proposée.")
        return valeur


class VariableNumerique(Variable):
    """Vérifie qu'une valeur numérique est dans l'intervalle attendu."""

    def __init__(self, nom: str, minimum: float, maximum: float):
        super().__init__(nom)
        self._minimum = minimum
        self._maximum = maximum

    def valider(self, valeur) -> float:
        try:
            nombre = float(valeur)
        except (TypeError, ValueError) as erreur:
            raise SaisieInvalideError(f"{self} : entre un nombre valide.") from erreur
        if not self._minimum <= nombre <= self._maximum:
            raise SaisieInvalideError(
                f"{self} : valeur attendue entre {self._minimum:g} et {self._maximum:g}."
            )
        return nombre


class Application:
    """Regroupe le modèle prédictif et les règles de validation des entrées."""

    def __init__(self, modele, metadonnees: dict):
        self._modele = modele
        self._metadonnees = metadonnees
        self._variables = {}
        for nom, choix in metadonnees["categories"].items():
            self._variables[nom] = VariableCategorielle(nom, choix)
        for nom, limites in metadonnees["numeriques"].items():
            self._variables[nom] = VariableNumerique(nom, limites[0], limites[1])

    def predire(self, valeurs: dict) -> float:
        """Valide les valeurs, puis prédit une durée positive en heures."""
        saisie = {}
        for nom in self._metadonnees["colonnes"]:
            saisie[nom] = self._variables[nom].valider(valeurs.get(nom))
        tableau = pd.DataFrame([saisie], columns=self._metadonnees["colonnes"])
        return max(0.0, float(self._modele.predict(tableau)[0]))


def creer_metadonnees(projet: Modele) -> dict:
    """Décrit les variables à saisir selon les données d'entraînement."""
    donnees = projet.X_train
    bornes = {
        nom: [float(donnees[nom].min()), float(donnees[nom].max())]
        for nom in projet.numeric_features
    }
    # Les valeurs du calendrier restent valides même hors de la courte période étudiée.
    bornes.update({
        "Created Month": [1, 12], "Created Day": [1, 31],
        "Created Hour": [0, 23], "Created Minute": [0, 59],
    })
    return {
        "colonnes": projet.features,
        "categories": {
            nom: sorted(donnees[nom].dropna().unique().tolist())
            for nom in projet.categorical_features
        },
        "numeriques": bornes,
    }


@st.cache_resource
def charger_application():
    """Charge le modèle enregistré ou l'entraîne une seule fois si nécessaire."""
    projet = Modele()
    os.makedirs(DOSSIER, exist_ok=True)

    if os.path.isfile(FICHIER_MODELE) and os.path.isfile(FICHIER_META):
        modele = joblib.load(FICHIER_MODELE)
        with open(FICHIER_META, "r", encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)
    else:
        modele = Pipeline([
            ("preprocessing", projet.preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            )),
        ])
        modele.fit(projet.X_train, projet.y_train)
        metadonnees = creer_metadonnees(projet)
        joblib.dump(modele, FICHIER_MODELE)
        with open(FICHIER_META, "w", encoding="utf-8") as fichier:
            json.dump(metadonnees, fichier, ensure_ascii=False, indent=2)

    predictions = modele.predict(projet.X_test)
    mae = mean_absolute_error(projet.y_test, predictions)
    reference = projet.baseline()
    mae_reference = mean_absolute_error(projet.y_test, reference)
    return projet, Application(modele, metadonnees), metadonnees, mae, mae_reference


def enregistrer_prediction(valeurs: dict, duree: float) -> None:
    """Ajoute une prédiction dans un fichier CSV consultable dans l'interface."""
    os.makedirs(DOSSIER, exist_ok=True)
    existe = os.path.isfile(FICHIER_HISTORIQUE)
    with open(FICHIER_HISTORIQUE, "a", newline="", encoding="utf-8") as fichier:
        colonnes = ["Date de prédiction", *valeurs.keys(), "Durée estimée (heures)"]
        ecrivain = csv.DictWriter(fichier, fieldnames=colonnes)
        if not existe or os.path.getsize(FICHIER_HISTORIQUE) == 0:
            ecrivain.writeheader()
        ecrivain.writerow({
            "Date de prédiction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **valeurs,
            "Durée estimée (heures)": round(duree, 2),
        })


def afficher_graphique(mae: float, reference: float) -> None:
    """Affiche avec Matplotlib la comparaison des erreurs du modèle."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(["Référence", "Random Forest"], [reference, mae])
    ax.set_title("Erreur moyenne des modèles")
    ax.set_xlabel("MAE (heures) — plus petit = meilleur")
    ax.set_ylabel("Modèle")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    """Affiche les saisies, la prédiction, le graphique et l'historique."""
    st.set_page_config(page_title="NYC 311 — Prédiction", page_icon="🏙️")
    st.title("🏙️ NYC 311 — Durée de résolution")
    st.write("Estimation du temps de fermeture d'une plainte à New York.")
    st.caption("Projet pédagogique : ce résultat n'est pas un délai officiel.")

    try:
        with st.spinner("Chargement du modèle..."):
            projet, application, meta, mae, reference = charger_application()
    except (OSError, ValueError, KeyError) as erreur:
        st.error(f"Impossible de charger le modèle : {erreur}")
        st.stop()

    donnees = projet.X_train
    st.subheader("Informations sur la plainte")
    # Hors du formulaire : changer de problème actualise immédiatement les agences.
    problemes = meta["categories"][COLONNE_PROBLEME]
    probleme = st.selectbox("Type de problème", problemes)
    agences = sorted(donnees.loc[donnees[COLONNE_PROBLEME] == probleme, "Agency"].unique())
    agence = st.selectbox("Agence responsable", agences)
    with st.form("plainte"):
        arrondissement = st.selectbox("Arrondissement", meta["categories"]["Borough"])
        date = st.date_input("Date du signalement")
        heure = st.time_input("Heure du signalement")
        envoyer = st.form_submit_button("Prédire la durée", type="primary")

    zone = donnees[donnees["Borough"] == arrondissement]
    valeurs = {
        "Agency": agence,
        COLONNE_PROBLEME: probleme,
        "Borough": arrondissement,
        "Latitude": float(zone["Latitude"].median()),
        "Longitude": float(zone["Longitude"].median()),
        "Created Month": date.month,
        "Created Day": date.day,
        "Created Hour": heure.hour,
        "Created Minute": heure.minute,
    }
    st.caption("Coordonnées approximées automatiquement selon l'arrondissement.")

    if envoyer:
        try:
            duree = application.predire(valeurs)
            enregistrer_prediction(valeurs, duree)
        except (SaisieInvalideError, OSError, ValueError) as erreur:
            st.error(str(erreur))
        else:
            st.success("Prédiction effectuée")
            a, b = st.columns(2)
            a.metric("Durée estimée", f"{duree:.1f} heures")
            b.metric("En jours", f"{duree / 24:.1f} jours")
            st.info("Estimation statistique : la durée réelle peut être différente.")

    st.subheader("Performance du modèle")
    st.write(f"Random Forest : **{mae:.1f} h** d'erreur moyenne ; référence : **{reference:.1f} h**.")
    afficher_graphique(mae, reference)

    st.subheader("Historique des prédictions")
    if os.path.isfile(FICHIER_HISTORIQUE):
        try:
            historique = pd.read_csv(FICHIER_HISTORIQUE)
            st.dataframe(historique, use_container_width=True)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as erreur:
            st.warning(f"Historique indisponible : {erreur}")
    else:
        st.info("Aucune prédiction enregistrée pour l'instant.")


if __name__ == "__main__":
    main()
