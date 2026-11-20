"""Interface Streamlit du projet INF1007 : prédiction du délai des plaintes NYC 311.

L'interface utilise le module modele.py du binôme, sans le modifier.
L'entraînement n'a lieu que si aucun modèle sauvegardé n'est disponible.
"""

import csv
import json
import math
import os
from abc import ABC, abstractmethod
from datetime import date, datetime, time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modele import Modele


RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_MODELES = os.path.join(RACINE, "modeles")
FICHIER_MODELE = os.path.join(DOSSIER_MODELES, "modele.pk1")
FICHIER_META = os.path.join(DOSSIER_MODELES, "metadonnees.json")
FICHIER_HISTORIQUE = os.path.join(DOSSIER_MODELES, "historique_predictions.csv")
FICHIER_HISTORIQUE_V2 = os.path.join(DOSSIER_MODELES, "historique_predictions_v2.csv")
FICHIER_DONNEES = os.path.join(RACINE, "donnees", "donnees.csv")
PROBLEME = "Problem (formerly Complaint Type)"
CATEGORIELLES = [PROBLEME, "Agency", "Borough"]
NUMERIQUES = ["Latitude", "Longitude", "Created Month", "Created Day",
              "Created Hour", "Created Minute"]
COLONNES_HISTORIQUE = ["Date de prédiction", "Date du signalement", "Agency", PROBLEME, "Borough",
                       "Latitude", "Longitude", "Created Month", "Created Day",
                       "Created Hour", "Created Minute", "Durée estimée (heures)"]


NOMS_AGENCES: dict[str, str] = {
    "NYPD": "New York City Police Department (police)",
    "HPD": "Department of Housing Preservation and Development (logement)",
    "DSNY": "Department of Sanitation (propreté et déchets)",
    "DOT": "Department of Transportation (voirie et circulation)",
    "DEP": "Department of Environmental Protection (environnement)",
    "DPR": "Department of Parks and Recreation (parcs)",
    "DOB": "Department of Buildings (bâtiments)",
    "DOHMH": "Department of Health and Mental Hygiene (santé publique)",
    "DCWP": "Department of Consumer and Worker Protection (consommateurs et travailleurs)",
    "TLC": "Taxi and Limousine Commission (taxis et VTC)",
    "FDNY": "Fire Department of New York (pompiers)",
    "DHS": "Department of Homeless Services (aide aux sans-abri)",
    "HRA": "Human Resources Administration (aides sociales)",
    "DCA": "Department of Consumer Affairs (ancienne dénomination de DCWP)",
}


def libelle_agence(code: str) -> str:
    """Affiche un nom explicite tout en gardant le code original pour le modèle."""
    description: str | None = NOMS_AGENCES.get(code)
    return f"{code} — {description}" if description else code


class SaisieInvalideError(ValueError):
    """Erreur levée lorsqu'une caractéristique n'est pas autorisée."""


class Variable(ABC):
    """Classe parente des caractéristiques du modèle."""

    def __init__(self, nom: str):
        self._nom = nom

    def __str__(self) -> str:
        return self._nom

    @abstractmethod
    def valider(self, valeur: object) -> object:
        """Retourne une valeur valide, sinon lève SaisieInvalideError."""


class VariableCategorielle(Variable):
    """Caractéristique choisie dans une liste de catégories."""

    def __init__(self, nom: str, choix: list[str]):
        super().__init__(nom)
        self._choix = choix

    def valider(self, valeur: object) -> str:
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

    def valider(self, valeur: object) -> float:
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

    def __init__(self, modele: object, metadonnees: dict) -> None:
        self._modele = modele
        self._metadonnees = metadonnees
        self._variables = {}
        for nom, choix in metadonnees["categories"].items():
            self._variables[nom] = VariableCategorielle(nom, choix)
        for nom, limites in metadonnees["numeriques"].items():
            self._variables[nom] = VariableNumerique(nom, limites[0], limites[1])

    def predire(self, valeurs: dict[str, object]) -> float:
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


def verifier_metadonnees(meta: dict) -> None:
    """Vérifie que le JSON décrit toutes les variables attendues."""
    if not all(cle in meta for cle in ("colonnes", "categories", "numeriques")):
        raise ValueError("Le fichier de métadonnées est incomplet.")
    for nom in meta["colonnes"]:
        if nom not in meta["categories"] and nom not in meta["numeriques"]:
            raise ValueError(f"Métadonnées manquantes pour {nom}.")
    for nom in ("mae_heures", "mae_reference_heures"):
        if nom not in meta or not math.isfinite(float(meta[nom])):
            raise ValueError(f"Métrique manquante ou invalide : {nom}.")
    for nom in CATEGORIELLES:
        if nom not in meta["categories"] or not meta["categories"][nom]:
            raise ValueError(f"Catégories manquantes pour {nom}.")
    for nom in NUMERIQUES:
        bornes = meta["numeriques"].get(nom)
        if not isinstance(bornes, list) or len(bornes) != 2:
            raise ValueError(f"Intervalle manquant pour {nom}.")
        minimum, maximum = [float(x) for x in bornes]
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            raise ValueError(f"Intervalle invalide pour {nom}.")


def charger_projet() -> Modele:
    """Initialise le modèle du binôme depuis la racine du dépôt."""
    if not os.path.isfile(FICHIER_DONNEES):
        raise FileNotFoundError(f"Fichier de données introuvable : {FICHIER_DONNEES}")
    os.chdir(RACINE)
    return Modele()


def creer_metadonnees(projet: Modele, mae: float, mae_reference: float) -> dict[str, Any]:
    """Déduit les catégories et les coordonnées du jeu réellement chargé."""
    donnees: pd.DataFrame = projet.X
    categories: dict[str, list[str]] = {}
    for nom in CATEGORIELLES:
        categories[nom] = sorted({str(v) for v in donnees[nom].dropna().unique()
                                 if str(v).strip()})

    numeriques: dict[str, list[float]] = {
        "Latitude": [-90.0, 90.0], "Longitude": [-180.0, 180.0],
        "Created Month": [1.0, 12.0], "Created Day": [1.0, 31.0],
        "Created Hour": [0.0, 23.0], "Created Minute": [0.0, 59.0]
    }
    positions: pd.DataFrame = donnees[["Borough", "Latitude", "Longitude"]].copy()
    positions["Latitude"] = pd.to_numeric(positions["Latitude"], errors="coerce")
    positions["Longitude"] = pd.to_numeric(positions["Longitude"], errors="coerce")
    positions = positions.dropna(subset=["Latitude", "Longitude"])
    positions = positions.loc[positions["Latitude"].between(-90, 90)
                              & positions["Longitude"].between(-180, 180)]
    coordonnees: dict[str, list[float]] = {}
    for arrondissement, groupe in positions.groupby("Borough"):
        if pd.notna(arrondissement):
            coordonnees[str(arrondissement)] = [
                float(groupe["Latitude"].median()),
                float(groupe["Longitude"].median())
            ]

    return {"colonnes": list(projet.features), "categories": categories,
            "numeriques": numeriques, "coordonnees": coordonnees,
            "mae_heures": float(mae), "mae_reference_heures": float(mae_reference)}


@st.cache_resource
def charger_application() -> tuple[Application, dict[str, Any]]:
    """Charge les artefacts existants ou entraîne un équivalent du forest() du binôme.

    Modele.forest() renvoie les prédictions, pas l'estimateur entraîné.
    On recrée donc exactement son Pipeline ici pour pouvoir le sauvegarder,
    sans modifier src/modele.py.
    """
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    if os.path.isfile(FICHIER_MODELE) and os.path.isfile(FICHIER_META):
        modele = joblib.load(FICHIER_MODELE)
        with open(FICHIER_META, "r", encoding="utf-8") as fichier:
            meta = json.load(fichier)
        verifier_metadonnees(meta)
        return Application(modele, meta), meta

    projet: Modele = charger_projet()
    modele = Pipeline([
        ("preprocessing", projet.preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_split=2,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ))
    ])
    modele.fit(projet.X_train, projet.y_train)
    predictions = modele.predict(projet.X_test)
    mae = float(mean_absolute_error(projet.y_test, predictions))
    reference = DummyRegressor(strategy="median")
    reference.fit(projet.X_train, projet.y_train)
    mae_reference = float(mean_absolute_error(projet.y_test,
                                              reference.predict(projet.X_test)))
    meta = creer_metadonnees(projet, mae, mae_reference)
    verifier_metadonnees(meta)
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


@st.cache_data(show_spinner=False)
def agences_par_probleme(chemin: str, modification: float) -> dict[str, list[str]]:
    """Associe chaque problème aux agences observées dans le fichier 311."""
    associations: dict[str, set[str]] = {}
    for bloc in pd.read_csv(chemin, usecols=[PROBLEME, "Agency"], chunksize=50000):
        for probleme, agence in bloc[[PROBLEME, "Agency"]].dropna().itertuples(index=False, name=None):
            cle, valeur = str(probleme).strip(), str(agence).strip()
            if cle and valeur:
                associations.setdefault(cle, set()).add(valeur)
    return {probleme: sorted(agences) for probleme, agences in associations.items()}


def choisir_agences(meta: dict, probleme: str) -> tuple[list[str], bool]:
    """Ne propose que les agences présentes dans les données et dans le modèle."""
    autorisees = list(meta["categories"]["Agency"])
    if not os.path.isfile(FICHIER_DONNEES):
        return autorisees, False
    try:
        associations = agences_par_probleme(FICHIER_DONNEES, os.path.getmtime(FICHIER_DONNEES))
        correspondances = [agence for agence in autorisees
                           if agence in associations.get(probleme, [])]
        if correspondances:
            return correspondances, True
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        pass
    return autorisees, False


def lire_entete_historique(chemin: str) -> list[str]:
    """Lit l'en-tête sans modifier le fichier d'historique."""
    if not os.path.isfile(chemin) or os.path.getsize(chemin) == 0:
        return []
    with open(chemin, "r", newline="", encoding="utf-8-sig") as fichier:
        return next(csv.reader(fichier), [])


def choisir_fichier_historique() -> str:
    """Conserve l'ancien CSV et utilise un nouveau fichier si son format diffère."""
    entete = lire_entete_historique(FICHIER_HISTORIQUE)
    if not entete or entete == COLONNES_HISTORIQUE:
        return FICHIER_HISTORIQUE
    ancienne = [colonne for colonne in COLONNES_HISTORIQUE
               if colonne != "Date du signalement"]
    if entete != ancienne:
        raise ValueError("Format d'historique inconnu : aucun fichier n'a été modifié.")
    nouvelle_entete = lire_entete_historique(FICHIER_HISTORIQUE_V2)
    if nouvelle_entete and nouvelle_entete != COLONNES_HISTORIQUE:
        raise ValueError("Le fichier d'historique v2 a un format incompatible.")
    return FICHIER_HISTORIQUE_V2


def enregistrer_prediction(valeurs: dict[str, object], duree: float,
                            date_signalement: datetime) -> None:
    """Ajoute la prédiction sans renommer ni écraser un historique existant."""
    os.makedirs(DOSSIER_MODELES, exist_ok=True)
    chemin = choisir_fichier_historique()
    nouveau = not os.path.isfile(chemin) or os.path.getsize(chemin) == 0
    ligne: dict[str, object] = {
        "Date de prédiction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Date du signalement": date_signalement.strftime("%Y-%m-%d %H:%M"),
        **valeurs,
        "Durée estimée (heures)": round(duree, 2),
    }
    with open(chemin, "a", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES_HISTORIQUE)
        if nouveau:
            ecrivain.writeheader()
        ecrivain.writerow(ligne)


def afficher_graphique(meta: dict) -> None:
    """Compare les erreurs du Random Forest et du modèle de référence."""
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
    """Affiche une synthèse compacte, et garde tous les détails exportables."""
    st.subheader("Historique des prédictions")
    historiques: list[pd.DataFrame] = []
    anciens_formats: bool = False
    try:
        for chemin in (FICHIER_HISTORIQUE, FICHIER_HISTORIQUE_V2):
            if not os.path.isfile(chemin) or os.path.getsize(chemin) == 0:
                continue
            entete: list[str] = lire_entete_historique(chemin)
            ancienne: list[str] = [colonne for colonne in COLONNES_HISTORIQUE
                                   if colonne != "Date du signalement"]
            if entete not in (COLONNES_HISTORIQUE, ancienne):
                st.warning(f"Historique ignoré (format inattendu) : {os.path.basename(chemin)}")
                continue
            tableau: pd.DataFrame = pd.read_csv(chemin, encoding="utf-8-sig")
            if entete == ancienne:
                anciens_formats = True
                tableau["Date du signalement"] = ""
            historiques.append(tableau.reindex(columns=COLONNES_HISTORIQUE))

        if not historiques:
            st.info("Aucune prédiction enregistrée pour l'instant.")
            return

        historique: pd.DataFrame = pd.concat(historiques, ignore_index=True)
        historique = historique.sort_values("Date de prédiction", ascending=False,
                                            kind="stable").reset_index(drop=True)
        colonnes_essentielles: dict[str, str] = {
            "Date de prédiction": "Prédiction le",
            "Date du signalement": "Signalement le",
            PROBLEME: "Type de problème",
            "Agency": "Agence",
            "Borough": "Arrondissement",
            "Durée estimée (heures)": "Durée (h)",
        }
        resume: pd.DataFrame = historique[list(colonnes_essentielles)].rename(
            columns=colonnes_essentielles).copy()
        resume["Durée (h)"] = pd.to_numeric(resume["Durée (h)"], errors="coerce")
        st.dataframe(
            resume,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prédiction le": st.column_config.TextColumn(width="medium"),
                "Signalement le": st.column_config.TextColumn(width="medium"),
                "Type de problème": st.column_config.TextColumn(width="large"),
                "Agence": st.column_config.TextColumn(width="small"),
                "Arrondissement": st.column_config.TextColumn(width="medium"),
                "Durée (h)": st.column_config.NumberColumn(format="%.1f", width="small"),
            },
        )
        if anciens_formats:
            st.caption("Les anciennes prédictions ne contiennent pas la date complète du "
                       "signalement : elle ne peut pas être reconstituée avec certitude.")

        st.download_button(
            "Télécharger l'historique complet (CSV)",
            historique.to_csv(index=False).encode("utf-8-sig"),
            file_name="historique_predictions_complet.csv",
            mime="text/csv",
        )
    except (OSError, UnicodeError, pd.errors.ParserError,
            pd.errors.EmptyDataError, ValueError) as erreur:
        st.warning(f"Historique indisponible : {erreur}")


<<<<<<< HEAD
def changer_page(page: str) -> None:
    """Change de page sans effacer le résultat conservé en session."""
    st.session_state["page_active"] = page

=======
def main() -> None:
    """Affiche les champs issus du JSON, la prédiction et l'historique."""
    st.set_page_config(page_title="NYC 311 — Prédiction", page_icon="🏙️")
    st.title("🏙️ NYC 311 — Durée de résolution")
    st.write("Estimation du temps de fermeture d'une plainte à New York.")
    st.caption("Projet pédagogique : ce résultat n'est pas un délai officiel.")
    application, meta = charger_application()
    try:
        with st.spinner("Chargement du modèle..."):
            application, meta = charger_application()
    except (OSError, ValueError, KeyError, ImportError, AttributeError, json.JSONDecodeError) as erreur:
        st.error(f"Impossible de charger le modèle : {erreur}")
        st.stop()
>>>>>>> main

def page_prediction(application: Application, meta: dict) -> None:
    """Collecte les caractéristiques, puis enregistre la prédiction."""
    st.subheader("Informations sur la plainte")
    st.caption("Choisissez d'abord le problème : les agences proposées s'adaptent immédiatement.")
    probleme: str = st.selectbox("Type de problème", meta["categories"][PROBLEME])
    choix_agences, association_verifiee = choisir_agences(meta, probleme)
    agence: str = st.selectbox("Agence responsable", choix_agences,
                               format_func=libelle_agence, key=f"agence_{probleme}")
    if not association_verifiee:
        st.caption("Correspondance non vérifiée dans les données : toutes les agences compatibles sont proposées.")
    arrondissement: str = st.selectbox("Arrondissement", meta["categories"]["Borough"])
    with st.form("plainte"):
        date_signalement: date = st.date_input("Date du signalement", value=date.today())
        heure_signalement: time = st.time_input("Heure du signalement", value=time(12, 0), step=60)
        automatique: bool = st.checkbox("Coordonnées approximatives automatiques", value=True)
        if automatique:
            try:
                latitude, longitude = coordonnees_proposees(meta, arrondissement)
                st.caption(f"Position indicative : {latitude:.4f}, {longitude:.4f}.")
            except SaisieInvalideError as erreur:
                st.warning(str(erreur))
                latitude, longitude = None, None
        else:
            lat_min, lat_max = meta["numeriques"]["Latitude"]
            lon_min, lon_max = meta["numeriques"]["Longitude"]
            latitude = st.text_input("Latitude", help=f"Entre {lat_min:g} et {lat_max:g}")
            longitude = st.text_input("Longitude", help=f"Entre {lon_min:g} et {lon_max:g}")
        envoyer: bool = st.form_submit_button("Prédire la durée", type="primary")

    if not envoyer:
        return
    date_complete: datetime = datetime.combine(date_signalement, heure_signalement)
    valeurs: dict[str, object] = {
        "Agency": agence, PROBLEME: probleme, "Borough": arrondissement,
        "Latitude": latitude, "Longitude": longitude,
        "Created Month": date_complete.month, "Created Day": date_complete.day,
        "Created Hour": date_complete.hour, "Created Minute": date_complete.minute
    }
    try:
        duree: float = application.predire(valeurs)
    except (SaisieInvalideError, ValueError, TypeError) as erreur:
        st.error(str(erreur))
        return
    st.session_state["dernier_resultat"] = {
        "duree": duree, "date_signalement": date_complete.strftime("%Y-%m-%d %H:%M"),
        "probleme": probleme, "agence": agence, "arrondissement": arrondissement,
        "valeurs": valeurs
    }
    try:
        enregistrer_prediction(valeurs, duree, date_complete)
        st.session_state["alerte_historique"] = ""
    except (OSError, UnicodeError, ValueError, csv.Error) as erreur:
        st.session_state["alerte_historique"] = str(erreur)
    st.session_state["page_active"] = "Résultats"
    st.rerun()


def page_resultats(meta: dict) -> None:
    """Présente le dernier résultat et les performances sur une page indépendante."""
    st.subheader("Résultats")
    resultat: dict | None = st.session_state.get("dernier_resultat")
    if resultat is None:
        st.info("Effectuez une prédiction pour afficher son résultat ici.")
    else:
        st.success("Prédiction effectuée")
        a, b = st.columns(2)
        a.metric("Durée estimée", f"{resultat['duree']:.1f} heures")
        b.metric("En jours", f"{resultat['duree'] / 24:.1f} jours")
        st.write(f"**Signalement :** {resultat['date_signalement']}")
        st.write(f"**Problème :** {resultat['probleme']}")
        st.write(f"**Agence :** {libelle_agence(resultat['agence'])}")
        st.write(f"**Arrondissement :** {resultat['arrondissement']}")
        st.info("Estimation statistique : la durée réelle peut être différente.")
        if st.session_state.get("alerte_historique"):
            st.warning("Résultat calculé, mais historique non enregistré : " +
                       st.session_state["alerte_historique"])
    st.subheader("Performance du modèle")
    st.write(f"Random Forest : **{meta['mae_heures']:.1f} h** d'erreur moyenne ; "
             f"référence : **{meta['mae_reference_heures']:.1f} h**.")
    afficher_graphique(meta)


def main() -> None:
    """Gère la navigation entre prédiction, résultats et historique."""
    st.set_page_config(page_title="NYC 311 — Prédiction", page_icon="🏙️", layout="wide")
    st.title("🏙️ NYC 311 — Durée de résolution")
    st.caption("Projet pédagogique : ce résultat n'est pas un délai officiel.")
    if "page_active" not in st.session_state:
        st.session_state["page_active"] = "Prédiction"
    st.sidebar.title("Navigation")
    for page in ("Prédiction", "Résultats", "Historique"):
        st.sidebar.button(page, key=f"nav_{page}", use_container_width=True,
                          type="primary" if st.session_state["page_active"] == page else "secondary",
                          on_click=changer_page, args=(page,))
    try:
        with st.spinner("Chargement du modèle (premier lancement : entraînement possible)..."):
            application, meta = charger_application()
    except (OSError, ValueError, KeyError, ImportError, AttributeError, json.JSONDecodeError) as erreur:
        st.error(f"Impossible de charger le modèle : {erreur}")
        st.stop()
    page: str = st.session_state["page_active"]
    if page == "Prédiction":
        page_prediction(application, meta)
    elif page == "Résultats":
        page_resultats(meta)
    else:
        afficher_historique()


if __name__ == "__main__":
    main()
