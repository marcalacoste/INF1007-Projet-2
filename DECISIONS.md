### Volet 1

A) Nous avons choisi d'analyser les appels au 311 de New York, parce que j'ai déjà travaillé avec ce jeux de données dans le passé, donc je connais la quantité d'informations utiles qu'il contient

B) Nous avons choisi ce jeu de données parce qu'il contient beaucoup d'informations et chaque appel est bien catalogué et contient beaucoup de détails pertinents. Puisque le 311 reçoit des milliers d'appels par jour, nous avons limité les données aux 50 000 premiers appels de 2025. Ce jeu de données est public d'accès pour tout le monde

### Volet 2

En sauvegardant premièrement nos données dans une DataFrame de pandas en utilisant pandas.read_csv(), nous pouvons ensuite trouver enelver les rangées avec des valeurs manquantes avec Dataframe.drop(). Les seules données manquantes importantes sont les dates de fermetures de la plainte, donc nous avons simplement enlevé les lignes qui n'ont pas de date de fermeture. Nous avons choisi de seulement garder les colonnes Created Date, Closed Date, Agency, Problem (formerly Complaint Type) Borough, Latitude, Longitude pour entraîner le modèle, puisque les données initiales contiennent plus de 30 colonnes, dont la plupart sont vides, puisqu'elle ne sont pas nécessaire pour entraîner notre modèle et ne rajoutent pas beaucoup de contexte important. Finalement, nous avons ajouté une nouvelle colonne appelée Resolution Time (hours), qui représente le temps de résolution du problème en heures. Cette colonne est calculée à partir de Created Data et Closed Date. Certaines valeurs pour le temps de résolution étaient négatives, donc nous les avons retirées de nos données. Nous avons aussi ajouté les colonnes Created Month, Created Day, Created Hour et Created Minute au données, puisque le modèle à une plus grande facilité à travailler avec des chiffres au lieu d'une date.

Notre variable cible est le temps de résolution. C'est un problème de régression, puisque nous essayons de préduire un chiffre excat basé sur nos données. Comme métrique d'évaluation, nous utilisons MAE (Mean Absolute Error), qui représente l'erreur absolue moyenne entre les prédictions et les valeurs réelles. Nous avons choisi cette métrique puisqu'elle est facile à comprendre, le résultat de MAE est bien descriptif de la performance du modèle. Comme modèle trivial, nous avons fait la moyenne du temps de résolution, qui est de 224.34 heures.

TODO: Montrer erreurs sur données test, expliquer si et comment améliorer les erreurs. Aussi expliquer que l'erreur réduit beaucoup si on ignore les temps de réponse de plus de 1000 heures


### Volet 3
Nous avons choisi d'utiliser Streamlit pour développer notre interface graphique, car cette bibliothèque permet de créer facilement une application web en Python, avec des formulaires et des graphiques Matplotlib. Nous avions également envisagé Tkinter, mais Streamlit nous semblait plus adapté à notre projet. Notre objectif était de rendre l'interface aussi claire et intuitive que possible. Après validation avec le professeur, nous avons décidé de l'organiser en trois pages : Prédiction, Résultats et Historique, afin de faciliter la navigation.

Pour la saisie des informations, nous avons utilisé des listes déroulantes pour les variables catégorielles, comme le type de problème, l'agence et l'arrondissement, ainsi que des champs adaptés pour la date et l'heure. Nous avons également choisi de filtrer les agences selon le problème sélectionné pour éviter de proposer des choix inutiles.

Une difficulté concernait la latitude et la longitude, que l'utilisateur devait initialement entrer manuellement. Comme cette solution n'était pas très pratique, nous avons ajouté une option proposant des coordonnées approximatives selon l'arrondissement choisi, calculées à partir des données historiques. Nous avons tout de même conservé la saisie manuelle pour les utilisateurs qui connaissent leurs coordonnées.

Nous avons également utilisé `st.session_state` pour conserver la dernière prédiction lorsque l'utilisateur change de page, et `st.cache_resource` pour éviter de recharger inutilement le modèle à chaque interaction. Ces deux fonctionnalités permettent de rendre l'application plus pratique et plus fluide.

Pour présenter les résultats, nous avons choisi d'afficher la durée estimée en heures et en jours, accompagnée d'un graphique Matplotlib représentant les durées historiques de résolution pour le même type de problème. Une ligne verticale indique la prédiction pour permettre à l'utilisateur de la situer par rapport aux anciennes plaintes. Nous avons préféré cette visualisation à une simple comparaison des erreurs des modèles, car elle est plus utile pour l'utilisateur. Nous précisons également que le résultat reste une estimation et non un délai garanti.

Enfin, nous avons décidé d'enregistrer les prédictions dans un fichier CSV plutôt que de les conserver uniquement en mémoire. L'utilisateur peut ainsi retrouver ses anciennes prédictions dans la page Historique et télécharger le fichier. Ce choix permet de conserver les résultats même après la fermeture de l'application.