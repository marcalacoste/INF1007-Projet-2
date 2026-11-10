### Volet 1

A) Nous avons choisi d'analyser les appels au 311 de New York, parce que j'ai déjà travaillé avec ce jeux de données dans le passé, donc je connais la quantité d'informations utiles qu'il contient

B) Nous avons choisi ce jeu do données parce qu'il contient beaucoup d'informations et chaque appel est bien catalogué et contient beaucoup de détails pertinents. Puisque le 311 reçoit des milliers d'appels par jour, nous avons limité les données aux 50 000 premiers appels de 2025. Ce jeu de données est public d'accès pour tout le monde

### Volet 2

En sauvegardant premièrement nos données dans une DataFrame de pandas en utilisant pandas.read_csv(), nous pouvons ensuite trouver toutes les données manquantes en utilisant DataFrame.isnull().sum(), qui nous donne une quantité de valeurs manquantes pour chaque colonne du jeu de données. Plusieurs données sont vides, mais la plupart ne sont pas nécessaires pour entraîner le modèle. Les seules données manquantes importantes sont les dates de fermetures de la plainte, donc nous avons simplement enlevé les lignes qui n'ont pas de date de fermeture

