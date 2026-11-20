from donnees import JeuDeDonnees
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import json
import joblib
import os
import numpy as np

class Modele:
    def __init__(self):
        self.donnees = JeuDeDonnees(os.path.join("donnees", "donnees.csv"))
        self.donnees.setup_donnees(50000)

        self.features = [
            "Agency",
            "Problem (formerly Complaint Type)",
            "Borough",
            "Latitude",
            "Longitude",
            "Created Month",
            "Created Day",
            "Created Hour",
            "Created Minute"
        ]
        self.target = "Resolution Time (hours)"

        self.X = self.donnees.data[self.features]
        self.y = self.donnees.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

        self.numeric_features = [
            "Latitude",
            "Longitude",
            "Created Month",
            "Created Day",
            "Created Hour",
            "Created Minute"
        ]

        self.categorical_features = [
            "Problem (formerly Complaint Type)",
            "Borough",
            "Agency"
        ]

        self.numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
        self.categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
        self.preprocessor = ColumnTransformer([("numeric", self.numeric_pipeline,self. numeric_features), ("categorical", self.categorical_pipeline, self.categorical_features)])

        self.modele = None
        self.predictions = None

    def baseline(self):
        """Retourne les prédictions du modèle trivial (moyenne)"""
        model = DummyRegressor(strategy="mean")
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        return predictions

    def linear(self):
        """Retourne les prédictions d'une régression linéaire"""
        model = Pipeline([("preprocessing", self.preprocessor), ("regressor", LinearRegression())])
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.modele = model
        return predictions


    def forest(self):
        """Retourne les prédictions d'une forêt aléatoire"""
        model = Pipeline([
            ("preprocessing", self.preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.modele = model
        self.predictions = predictions
        return predictions

    def mae(self, predictions) -> float:
        """Retourne la Mean Absolute Error (MAE)"""
        return mean_absolute_error(self.y_test, predictions)

    def sauvegarder_modele(self, force: bool = False):
        """Sauvegarde les prédictions du modèle en utilisant joblib
        
        Ne va pas sauvegarder si le fichier existe déjà, à moins que force == True"""
        path = os.path.join("modeles", "modele.pk1")
        if force or not os.path.exists(path):
            joblib.dump(self.modele, path)
        

    def charger_modele(self):
        """Charge les prédictions du modèles si elles ont déjà été sauvegardées"""
        path = os.path.join("modeles", "modele.pk1")
        if os.path.exists(path):
            self.modele = joblib.load(path)
            self.predictions = self.modele.predict(self.X_test)
            return self.modele
        else:
            raise ValueError("Il n'y a pas de modèle à charger dans le dossier modeles")

    def creer_metadonnees(self, fichier: str, force: bool = False):
        """Crée le fichier de metadonnées
        
        Ne va pas sauvegarder si le fichier existe déjà, à moins que force == True"""
        if force or not os.path.exists(fichier):
            colonnes = list(self.donnees.data.columns)[2:]
            # On enlève la colonne Resolution Time (hours)
            index = colonnes.index("Resolution Time (hours)")
            colonnes = colonnes[:index]+colonnes[index + 1:]
            problem = list(np.sort(self.donnees.data["Problem (formerly Complaint Type)"].unique()))
            borough = list(np.sort(self.donnees.data["Borough"].unique()))
            agency = list(np.sort(self.donnees.data["Agency"].unique()))
            latitude = [float(self.donnees.data["Latitude"].min()), float(self.donnees.data["Latitude"].max())]
            longitude = [float(self.donnees.data["Longitude"].min()), float(self.donnees.data["Longitude"].max())]
            groupes = self.donnees.data.groupby("Borough")
            lon_avg = groupes["Longitude"].median()
            lat_avg = groupes["Latitude"].median()

            meta = {
                "colonnes": colonnes,
                "categories": {
                    "Problem (formerly Complaint Type)": problem,
                    "Borough": borough,
                    "Agency": agency
                },
                "numeriques": {
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "Created Month": [1, 12],
                    "Created Day": [1,31],
                    "Created Hour": [0, 23],
                    "Created Minute": [0, 59]
                },
                "coordonnees": {
                    "BRONX": [lat_avg["BRONX"], lon_avg["BRONX"]],
                    "BROOKLYN": [lat_avg["BROOKLYN"], lon_avg["BROOKLYN"]],
                    "MANHATTAN": [lat_avg["MANHATTAN"], lon_avg["MANHATTAN"]],
                    "QUEENS": [lat_avg["QUEENS"], lon_avg["QUEENS"]],
                    "STATEN ISLAND": [lat_avg["STATEN ISLAND"], lon_avg["STATEN ISLAND"]],
                    "Unspecified": [lat_avg["Unspecified"], lon_avg["Unspecified"]]
                },
                "mae_heures": self.mae(self.predictions),
                "mae_reference_heures": self.mae(self.baseline())
            }

            with open(fichier, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
    
