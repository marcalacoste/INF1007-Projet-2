from donnees import JeuDeDonnees
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

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

        self.numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        self.categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
        self.preprocessor = ColumnTransformer([("numeric", self.numeric_pipeline,self. numeric_features), ("categorical", self.categorical_pipeline, self.categorical_features)])

    def baseline(self):
        model = DummyRegressor(strategy="median")
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        return predictions

    def linear(self):
        model = Pipeline([("preprocessing", self.preprocessor), ("regressor", LinearRegression())])
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        return predictions


    def forest(self):
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
        return predictions

    def mae(self, predictions):
        return mean_absolute_error(self.y_test, predictions)
    
modele = Modele()
predictions = modele.forest()
print(modele.mae(predictions))
print(modele.donnees.moyenne())
