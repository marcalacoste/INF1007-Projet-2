### Volet 1

A) Nous avons choisi d'analyser les appels au 311 de New York, parce que j'ai déjà travaillé avec ce jeux de données dans le passé, donc je connais la quantité d'informations utiles qu'il contient

B) Nous avons choisi ce jeu do données parce qu'il contient beaucoup d'informations et chaque appel est bien catalogué et contient beaucoup de détails pertinents. Puisque le 311 reçoit des milliers d'appels par jour, nous avons limité les données aux 50 000 premiers appels de 2025. Ce jeu de données est public d'accès pour tout le monde

### Volet 2

En sauvegardant premièrement nos données dans une DataFrame de pandas en utilisant pandas.read_csv(), nous pouvons ensuite trouver enelver les rangées avec des valeurs manquantes avec Dataframe.drop(). Les seules données manquantes importantes sont les dates de fermetures de la plainte, donc nous avons simplement enlevé les lignes qui n'ont pas de date de fermeture. Nous avons choisi de seulement garder les colonnes Created Date, Closed Date, Agency, Problem (formerly Complaint Type) Borough, Latitude, Longitude pour entraîner le modèle, puisque les données initiales contiennent plus de 30 colonnes, dont la plupart sont vides, puisqu'elle ne sont pas nécessaire pour entraîner notre modèle et ne rajoutent pas beaucoup de contexte important. Finalement, nous avons ajouté une nouvelle colonne appelée Resolution Time (hours), qui représente le temps de résolution du problème en heures. Cette colonne est calculée à partir de Created Data et Closed Date. Certaines valeurs pour le temps de résolution étaient négatives, donc nous les avons retirées de nos données. Nous avons aussi ajouté les colonnes Created Month, Created Day, Created Hour et Created Minute au données, puisque le modèle à une plus grande facilité à travailler avec des chiffres au lieu d'une date.

Notre variable cible est le temps de résolution. C'est un problème de régression, puisque nous essayons de préduire un chiffre excat basé sur nos données. Comme métrique d'évaluation, nous utilisons MAE (Mean Absolute Error), qui représente l'erreur absolue moyenne entre les prédictions et les valeurs réelles. Nous avons choisi cette métrique puisqu'elle est facile à comprendre, le résultat de MAE est bien descriptif de la performance du modèle. Comme modèle trivial, nous avons fait la moyenne du temps de résolution, qui est de 224.34 heures.


### Volet 3