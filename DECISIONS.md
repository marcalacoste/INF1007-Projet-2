### Volet 1

A) Nous avons choisi d'analyser les appels au 311 de New York, parce que j'ai déjà travaillé avec ce jeux de données dans le passé, donc je connais la quantité d'informations utiles qu'il contient

B) Nous avons choisi ce jeu do données parce qu'il contient beaucoup d'informations et chaque appel est bien catalogué et contient beaucoup de détails pertinents. Puisque le 311 reçoit des milliers d'appels par jour, nous avons limité les données aux 50 000 premiers appels de 2025. Ce jeu de données est public d'accès pour tout le monde

### Volet 2

En sauvegardant premièrement nos données dans une DataFrame de pandas en utilisant pandas.read_csv(), nous pouvons ensuite trouver enelver les rangées avec des valeurs manquantes avec Dataframe.drop(). Les seules données manquantes importantes sont les dates de fermetures de la plainte, donc nous avons simplement enlevé les lignes qui n'ont pas de date de fermeture. Nous avons choisi de seulement garder les colonnes Created Date, Closed Date, Agency, Problem (formerly Complaint Type) Borough, Latitude, Longitude pour entraîner le modèle, puisque les données initiales contiennent plus de 30 colonnes, dont la plupart sont vides, puisqu'elle ne sont pas nécessaire pour entraîner notre modèle et ne rajoutent pas beaucoup de contexte important.
