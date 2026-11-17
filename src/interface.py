"""Interface Streamlit du projet INF1007 : prédiction du délai des plaintes NYC 311.

L'interface utilise le module modele.py du binôme, sans le modifier.
L'entraînement n'a lieu que si aucun modèle sauvegardé n'est disponible.
"""

import csv
import json
import math
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


RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_MODELES = os.path.join(RACINE, "modeles")
FICHIER_MODELE = os.path.join(DOSSIER_MODELES, "modele_311.joblib")
FICHIER_META = os.path.join(DOSSIER_MODELES, "metadonnees.json")
FICHIER_HISTORIQUE = os.path.join(DOSSIER_MODELES, "historique_predictions.csv")
FICHIER_DONNEES = os.path.join(RACINE, "donnees", "donnees.csv")
PROBLEME = "Problem (formerly Complaint Type)"
CATEGORIELLES = [PROBLEME, "Agency", "Borough"]
NUMERIQUES = ["Latitude", "Longitude", "Created Month", "Created Day",
              "Created Hour", "Created Minute"]
COLONNES_HISTORIQUE = ["Date de prédiction", "Agency", PROBLEME, "Borough",
                       "Latitude", "Longitude", "Created Month", "Created Day",
                       "Created Hour", "Created Minute", "Durée estimée (heures)"]


class SaisieInvalideError(ValueError):
    """Erreur levée lorsqu'une caractéristique n'est pas autorisée."""


class Variable(ABC):
    """Classe parente des caractéristiques du modèle."""

    def __init__(self, nom: str):
        self._nom = nom

    def __str__(self) -> str:
        return self._nom

    @abstractmethod
    def valider(self, valeur):
        """Retourne une valeur valide, sinon lève SaisieInvalideError."""


class VariableCategorielle(Variable):
    """Caractéristique choisie dans une liste de catégories."""

    def __init__(self, nom: str, choix: list[str]):
        super().__init__(nom)
        self._choix = choix

    def valider(self, valeur) -> str:
        """Refuse une catégorie absente de la liste autorisée."""
        if valeur is None or valeur not in self._choix:
            raise SaisieInvalideError(f"{self} : sélectionne une catégorie proposée.")
        return str(valeur)


class VariableNumerique(Variable):
    """Caractéristique numérique limitée à un intervalle."""

    def __init__(self, nom: str, minimum: float, maximum: float):
        super().__init__(nom)
        self._minimum = float(minimum)
        self._maximum = float(maximum)

    def valider(self, valeur) -> float:
        """Convertit et valide un nombre fini dans l'intervalle autorisé."""
        try:
            if valeur is None or str(valeur).strip() == "":
                raise ValueError("valeur vide")
            nombre = float(valeur)
        except (TypeError, ValueError) as erreur:
            raise SaisieInvalideError(f"{self} : entre un nombre valide.") from erreur
        if not math.isfinite(nombre):
            raise SaisieInvalideError(f"{self} : le nombre doit être fini.")
        if not self._minimum <= nombre <= self._maximum:
            raise SaisieInvalideError(
                f"{self} : valeur attendue entre {self._minimum:g} et {self._maximum:g}."
            )
        return nombre


class Application:
    """Contient le modèle et les objets qui valident chaque caractéristique."""

    def __init__(self, modele, metadonnees: dict):
        self._modele = modele
        self._metadonnees = metadonnees
        self._variables = {}
        for nom, choix in metadonnees["categories"].items():
            self._variables[nom] = VariableCategorielle(nom, choix)
        for nom, limites in metadonnees["numeriques"].items():
            self._variables[nom] = VariableNumerique(nom, limites[0], limites[1])

    def predire(self, valeurs: dict) -> float:
        """Valide toutes les variables et prédit une durée non négative en heures."""
        saisie = {}
        for nom in self._metadonnees["colonnes"]:
            if nom not in self._variables:
                raise SaisieInvalideError(f"Métadonnées incomplètes pour {nom}.")
            saisie[nom] = self._variables[nom].valider(valeurs.get(nom))
        tableau = pd.DataFrame([saisie], columns=self._metadonnees["colonnes"])
        resultat = float(self._modele.predict(tableau)[0])
        if not math.isfinite(resultat):
            raise SaisieInvalideError("Le modèle a produit un résultat invalide.")
        return max(0.0, resultat)


def creer_metadonnees(projet: Modele, mae: float, reference: float) -> dict:
    """Produit les catégories, bornes, statistiques et métriques du modèle."""
    x = projet.X_train
    bornes = {nom: [float(x[nom].min()), float(x[nom].max())]
              for nom in projet.numeric_features}
    bornes.update({"Created Month": [1, 12], "Created Day": [1, 31],
                   "Created Hour": [0, 23], "Created Minute": [0, 59]})
    categories = {nom: sorted(x[nom].dropna().unique().tolist())
                  for nom in projet.categorical_features}
    coordonnees = {}
    for arrondissement, groupe in x.groupby("Borough"):
        coordonnees[str(arrondissement)] = [
            float(groupe["Latitude"].median()), float(groupe["Longitude"].median())
        ]
    return {"colonnes": list(projet.features), "categories": categories,
            "numeriques": bornes, "coordonnees": coordonnees,
            "mae_heures": float(mae), "mae_reference_heures": float(reference)}


def verifier_metadonnees(meta: dict) -> None:
    """Vérifie que le JSON décrit toutes les variables attendues."""
    if not all(cle in meta for cle in ("colonnes", "categories", "numeriques")):
        raise ValueError("Le fichier de métadonnées est incomplet.")
    for nom in meta["colonnes"]:
        if nom not in meta["categories"] and nom not in meta["numeriques"]:
            raise ValueError(f"Métadonnées manquantes pour {nom}.")
    for nom in CATEGORIELLES:
        if nom not in meta["categories"] or not meta["categories"][nom]:
            raise ValueError(f"Catégories manquantes pour {nom}.")
    for nom in NUMERIQUES:
        if nom not in meta["numeriques"] or len(meta["numeriques"][nom]) != 2:
            raise ValueError(f"Intervalle manquant pour {nom}.")


def charger_projet() -> Modele:
    """Charge les données de cours avec le module du binôme si elles existent."""
    if not os.path.isfile(FICHIER_DONNEES):
        raise FileNotFoundError(f"Fichier de données introuvable : {FICHIER_DONNEES}")
    # modele.py utilise actuellement un chemin relatif depuis la racine du dépôt.
    os.chdir(RACINE)
    return Modele()


@st.cache_resource
def charger_application():
    """Recharge le modèle sauvegardé, ou l'entraîne à la première utilisation."""
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    existe_modele = os.path.isfile(FICHIER_MODELE)
    existe_meta = os.path.isfile(FICHIER_META)
    if existe_modele and existe_meta:
        modele = joblib.load(FICHIER_MODELE)
        with open(FICHIER_META, "r", encoding="utf-8") as fichier:
            meta = json.load(fichier)
        verifier_metadonnees(meta)
        # Compatibilité avec les métadonnées de la première version du binôme.
        if not all(cle in meta for cle in ("coordonnees", "mae_heures", "mae_reference_heures")):
            projet = charger_projet()
            mae = mean_absolute_error(projet.y_test, modele.predict(projet.X_test))
            reference = mean_absolute_error(projet.y_test, projet.baseline())
            meta.update(creer_metadonnees(projet, mae, reference))
            with open(FICHIER_META, "w", encoding="utf-8") as fichier:
                json.dump(meta, fichier, ensure_ascii=False, indent=2)
    else:
        projet = charger_projet()
        modele = Pipeline([
            ("preprocessing", projet.preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=200, max_depth=None, min_samples_split=2,
                min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ])
        modele.fit(projet.X_train, projet.y_train)
        mae = mean_absolute_error(projet.y_test, modele.predict(projet.X_test))
        reference = mean_absolute_error(projet.y_test, projet.baseline())
        meta = creer_metadonnees(projet, mae, reference)
        joblib.dump(modele, FICHIER_MODELE)
        with open(FICHIER_META, "w", encoding="utf-8") as fichier:
            json.dump(meta, fichier, ensure_ascii=False, indent=2)
    return Application(modele, meta), meta


def coordonnees_proposees(meta: dict, arrondissement: str) -> tuple[float, float]:
    """Retourne la position médiane historique, ou celle du jeu complet."""
    valeurs = meta.get("coordonnees", {}).get(arrondissement)
    if valeurs is None:
        toutes = list(meta.get("coordonnees", {}).values())
        if not toutes:
            raise SaisieInvalideError("Coordonnées indisponibles pour cet arrondissement.")
        # Le cas « Unspecified » utilise un centre approximatif du jeu de données.
        valeurs = [sum(position[i] for position in toutes) / len(toutes)
                   for i in (0, 1)]
    return float(valeurs[0]), float(valeurs[1])


def enregistrer_prediction(valeurs: dict, duree: float) -> None:
    """Ajoute une ligne au CSV, sans effacer l'historique déjà présent."""
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    existe = os.path.isfile(FICHIER_HISTORIQUE)
    if existe and os.path.getsize(FICHIER_HISTORIQUE) > 0:
        with open(FICHIER_HISTORIQUE, "r", newline="", encoding="utf-8") as fichier:
            entete = next(csv.reader(fichier), [])
        if entete != COLONNES_HISTORIQUE:
            raise ValueError("Format de l'historique incorrect : aucune ligne n'a été ajoutée.")
    nouveau = not existe or os.path.getsize(FICHIER_HISTORIQUE) == 0
    with open(FICHIER_HISTORIQUE, "a", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES_HISTORIQUE)
        if nouveau:
            ecrivain.writeheader()
        ecrivain.writerow({"Date de prédiction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           **valeurs, "Durée estimée (heures)": round(duree, 2)})


def afficher_graphique(meta: dict) -> None:
    """Présente les erreurs MAE du modèle et de la référence avec Matplotlib."""
    mae = float(meta["mae_heures"])
    reference = float(meta["mae_reference_heures"])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(["Référence", "Random Forest"], [reference, mae],
            label="Erreur moyenne absolue")
    ax.set_title("Erreur moyenne des modèles")
    ax.set_xlabel("MAE (heures) — plus petit = meilleur")
    ax.set_ylabel("Modèle")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def afficher_historique() -> None:
    """Affiche les prédictions précédemment enregistrées dans le CSV."""
    st.subheader("Historique des prédictions")
    if not os.path.isfile(FICHIER_HISTORIQUE):
        st.info("Aucune prédiction enregistrée pour l'instant.")
        return
    try:
        historique = pd.read_csv(FICHIER_HISTORIQUE)
        st.dataframe(historique, use_container_width=True)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as erreur:
        st.warning(f"Historique indisponible : {erreur}")


def main() -> None:
    """Affiche les champs issus du JSON, la prédiction et l'historique."""
    st.set_page_config(page_title="NYC 311 — Prédiction", page_icon="🏙️")
    st.title("🏙️ NYC 311 — Durée de résolution")
    st.write("Estimation du temps de fermeture d'une plainte à New York.")
    st.caption("Projet pédagogique : ce résultat n'est pas un délai officiel.")

    try:
        with st.spinner("Chargement du modèle..."):
            application, meta = charger_application()
    except (OSError, ValueError, KeyError, ImportError, AttributeError, json.JSONDecodeError) as erreur:
        st.error(f"Impossible de charger le modèle : {erreur}")
        st.stop()

    st.subheader("Informations sur la plainte")
    with st.form("plainte"):
        probleme = st.selectbox("Type de problème", meta["categories"][PROBLEME])
        agence = st.selectbox("Agence responsable", meta["categories"]["Agency"])
        arrondissement = st.selectbox("Arrondissement", meta["categories"]["Borough"])
        date = st.date_input("Date du signalement")
        heure = st.time_input("Heure du signalement")
        automatique = st.checkbox("Coordonnées approximatives automatiques", value=True)
        if automatique:
            latitude, longitude = coordonnees_proposees(meta, arrondissement)
            st.caption(f"Position indicative : {latitude:.4f}, {longitude:.4f}. "
                       "Vous pouvez décocher pour saisir les coordonnées exactes.")
        else:
            lat_min, lat_max = meta["numeriques"]["Latitude"]
            lon_min, lon_max = meta["numeriques"]["Longitude"]
            latitude = st.text_input("Latitude", help=f"Entre {lat_min:g} et {lat_max:g}")
            longitude = st.text_input("Longitude", help=f"Entre {lon_min:g} et {lon_max:g}")
        envoyer = st.form_submit_button("Prédire la durée", type="primary")

    valeurs = {"Agency": agence, PROBLEME: probleme, "Borough": arrondissement,
               "Latitude": latitude, "Longitude": longitude,
               "Created Month": date.month, "Created Day": date.day,
               "Created Hour": heure.hour, "Created Minute": heure.minute}

    if envoyer:
        try:
            duree = application.predire(valeurs)
        except (SaisieInvalideError, ValueError, TypeError) as erreur:
            st.error(str(erreur))
        else:
            st.success("Prédiction effectuée")
            a, b = st.columns(2)
            a.metric("Durée estimée", f"{duree:.1f} heures")
            b.metric("En jours", f"{duree / 24:.1f} jours")
            st.info("Estimation statistique : la durée réelle peut être différente.")
            try:
                enregistrer_prediction(valeurs, duree)
            except (OSError, UnicodeError, ValueError, csv.Error) as erreur:
                st.warning(f"Prédiction affichée, mais historique non enregistré : {erreur}")

    st.subheader("Performance du modèle")
    st.write(f"Random Forest : **{meta['mae_heures']:.1f} h** d'erreur moyenne ; "
             f"référence : **{meta['mae_reference_heures']:.1f} h**.")
    afficher_graphique(meta)
    afficher_historique()


if __name__ == "__main__":
    main()
